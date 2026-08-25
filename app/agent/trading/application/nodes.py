"""Application nodes for the trading graph.

fundamentals (Phase 2), technical (Phase 3), news/sentiment (Phase 4), the
debate (Phase 5, in debate_nodes.py) and the risk panel (Phase 6, in
risk_nodes.py) are real. The synthesizer's own LLM call and citation
resolution live in infrastructure/synthesis_port.py — this module still owns
the caveat computation (`_news_caveats`, `_debate_caveats`, `_risk_caveats`),
same split as the risk/debate ports vs. their nodes.
"""
from collections import Counter
from datetime import date

from app.agent.trading.application.risk_ledger import build_risk_ledger
from app.agent.trading.domain.news_digest import (
    AGGREGATED_RELEVANCE,
    NewsDigest,
    NewsItem,
    SentimentSummary,
)
from app.agent.trading.domain.technical_report import TechnicalReport
from app.agent.trading.domain.trading_state import TradingState
from app.agent.trading.infrastructure.fundamentals_port import get_fundamentals_report
from app.agent.trading.infrastructure.news_data_port import fetch_company_news, filter_and_dedup
from app.agent.trading.infrastructure.news_digest_port import build_digest
from app.agent.trading.infrastructure.price_data_port import get_price_history
from app.agent.trading.application.technical_indicators import compute_indicators
from app.agent.trading.infrastructure.synthesis_port import run_synthesis
from app.agent.trading.infrastructure.technical_interpreter_port import interpret_indicators, save_technical_report


async def fundamentals_node(state: TradingState) -> dict:
    print(f"[fundamentals] running for {state['ticker']}")
    report = await get_fundamentals_report(state["ticker"])
    return {"fundamentals_report": report}


async def technical_node(state: TradingState) -> dict:
    ticker = state["ticker"]
    as_of = state.get("as_of_date")
    if as_of is None:
        # Same rule as news_node, for the same reason: a silent fallback to
        # "whatever the vendor considers current" is exactly how lookahead
        # contamination gets into a historical probe. See Gate C,
        # trading-agent-known-gaps.md item 16.
        raise ValueError(
            "as_of_date missing from TradingState — refusing to fetch price "
            "history unbounded. A technical fetch without an explicit upper "
            "bound is a lookahead bug."
        )
    print(f"[technical] running for {ticker} as of {as_of}")

    df, source, dropped_bars = await get_price_history(ticker, as_of)
    if dropped_bars:
        print(f"[technical] dropped {dropped_bars} incomplete bar(s) from {source}")
    indicators = compute_indicators(df)
    interpretation, flagged, flagged_claims, cost_usd = await interpret_indicators(
        ticker, indicators
    )
    if flagged_claims:
        print(f"[technical] {len(flagged_claims)} contradicted claim(s): {flagged_claims}")

    report = TechnicalReport(
        ticker=ticker,
        as_of_date=df.index[-1].date(),
        data_source=source,
        bars_used=len(df),
        bars_dropped_invalid=dropped_bars,
        indicators=indicators,
        interpretation=interpretation,
        interpretation_flagged_numbers=flagged,
        interpretation_flagged_claims=flagged_claims,
    )
    vault_path = save_technical_report(report, cost_usd=cost_usd)
    print(f"[technical] saved report to {vault_path}")
    return {"technical_report": report}


async def news_node(state: TradingState) -> dict:
    ticker = state["ticker"]
    as_of = state.get("as_of_date")
    if as_of is None:
        # Fail loud rather than defaulting to today: a silent
        # `or date.today()` fallback is precisely how lookahead
        # contamination gets into a backtest.
        raise ValueError(
            "as_of_date missing from TradingState — refusing to run unbounded. "
            "A news fetch without an explicit upper bound is a lookahead bug."
        )
    print(f"[news] running for {ticker} as of {as_of}")

    raw, window_start = await fetch_company_news(ticker, as_of)
    clean, dropped_win, dropped_missing, truncated = filter_and_dedup(
        raw, as_of, window_start
    )

    items, issues, cost_usd = await build_digest(clean, ticker)

    digest = NewsDigest(
        ticker=ticker,
        as_of_date=as_of,
        window_start=window_start,
        items=items,
        raw_article_count=len(raw),
        deduped_count=len(clean),
        dropped_out_of_window=dropped_win,
        dropped_missing_date=dropped_missing,
        truncated_by_cap=truncated,
    )

    # Belt-and-braces post-assertion. Cheap, and it turns a silent
    # correctness bug into a loud one at exactly the right moment.
    late = [i for i in digest.items if i.published_date > as_of]
    if late:
        raise AssertionError(
            f"Lookahead leak: {len(late)} article(s) dated after {as_of} "
            f"reached the digest — first: {late[0].published_date}"
        )

    print(
        f"[news] {len(items)} items (raw={len(raw)} deduped={len(clean)} "
        f"truncated={truncated}) cost={cost_usd}"
    )
    return {"news_digest": digest, "news_digest_issues": issues}


async def sentiment_node(state: TradingState) -> dict:
    """Deterministic aggregation over NewsDigest.items — no LLM, no network.

    Only articles whose relevance is in AGGREGATED_RELEVANCE are counted.
    The vendor feed tags broad sector coverage with the requested ticker, so
    aggregating everything measures the sector rather than the company.

    Two different situations both produce net_score=0.0 with
    article_count=0 — a genuinely quiet ticker, and a noisy feed where
    nothing was actually about the company — and neither is
    neutrality-with-evidence. `excluded_by_relevance` is what tells them
    apart downstream."""
    digest: NewsDigest = state["news_digest"]

    # A checkpoint written before a NewsItem field was added deserializes with
    # the outer NewsDigest intact but its items left as plain dicts — pydantic
    # cannot rebuild a child that is missing a now-required field, and nothing
    # raises at the read. Without this the first symptom is an AttributeError
    # on the new field, several frames from the actual cause. Same reason
    # news_node refuses to run without as_of_date: name the real problem at
    # the point it becomes knowable.
    stale = [i for i in digest.items if not isinstance(i, NewsItem)]
    if stale:
        raise TypeError(
            f"news_digest.items holds {len(stale)} {type(stale[0]).__name__} "
            f"entr{'y' if len(stale) == 1 else 'ies'} instead of NewsItem — this "
            f"checkpoint predates a NewsItem schema change and cannot be resumed. "
            f"Re-run the ticker under a new --thread-id."
        )

    relevant = [i for i in digest.items if i.relevance in AGGREGATED_RELEVANCE]
    excluded = len(digest.items) - len(relevant)
    print(
        f"[sentiment] aggregating {len(relevant)} of {len(digest.items)} items "
        f"for {digest.ticker} ({excluded} excluded by relevance)"
    )
    counts = Counter(i.sentiment for i in relevant)
    pos, neg, neu = counts["positive"], counts["negative"], counts["neutral"]
    total = pos + neg + neu
    return {
        "sentiment_summary": SentimentSummary(
            ticker=digest.ticker,
            as_of_date=digest.as_of_date,
            positive=pos,
            negative=neg,
            neutral=neu,
            net_score=(pos - neg) / total if total else 0.0,
            article_count=total,
            excluded_by_relevance=excluded,
        )
    }


# What each analyst leg is expected to leave behind in state. A partial run is
# a legitimate mode (--only), so a missing report is recorded as a data gap
# rather than raising — but the memo must never present a gap as a finding.
ANALYST_OUTPUTS = {
    "fundamentals": "fundamentals_report",
    "technical": "technical_report",
    "news": "news_digest",
}


def _news_caveats(state: TradingState) -> tuple[list[str], list[str]]:
    """What the news leg found, and what it could not see, as (gaps, evidence).

    The digest already records both; until this ran, neither reached the memo.
    A truncated digest saw a sample of the window and a reader acting on the
    memo alone had no way to know that — which is the whole reason
    `truncated_by_cap` exists rather than the cap being silent.
    """
    digest: NewsDigest | None = state.get("news_digest")
    if digest is None:
        return [], []

    gaps: list[str] = []
    evidence: list[str] = []

    if digest.truncated_by_cap:
        gaps.append(
            f"news digest is a SAMPLE, not the full window: the vendor returned "
            f"{digest.raw_article_count} articles for "
            f"{digest.window_start}..{digest.as_of_date} and the cap kept "
            f"{len(digest.items)}, newest first — coverage skews to the most "
            f"recent days and an older event may be absent entirely"
        )

    issues = state.get("news_digest_issues") or []
    if issues:
        gaps.append(
            f"{len(issues)} digest integrity issue(s) — article(s) may be missing "
            f"from the news evidence below: {'; '.join(issues[:3])}"
            + (f" (+{len(issues) - 3} more)" if len(issues) > 3 else "")
        )

    summary: SentimentSummary | None = state.get("sentiment_summary")
    if summary is None:
        return gaps, evidence

    if summary.article_count == 0:
        # Distinct from "news did not run", and distinct from neutral news.
        gaps.append(
            f"no articles in the window were primarily about {summary.ticker} "
            f"({summary.excluded_by_relevance} were excluded as not about it) — "
            f"the net sentiment score of 0.00 is an ABSENCE of evidence, not "
            f"evidence of neutrality, and must not be read as the latter"
        )
    else:
        evidence.append(
            f"news sentiment {summary.net_score:+.2f} over {summary.article_count} "
            f"article(s) primarily about {summary.ticker} "
            f"(+{summary.positive}/-{summary.negative}/={summary.neutral}) for "
            f"{digest.window_start}..{digest.as_of_date}; "
            f"{summary.excluded_by_relevance} further article(s) in the vendor "
            f"feed were excluded as not primarily about the company"
        )
        if summary.article_count < 5:
            gaps.append(
                f"news sentiment rests on only {summary.article_count} article(s); "
                f"a single item moves the score materially"
            )

    return gaps, evidence


def _debate_caveats(state: TradingState) -> tuple[list[str], list[str]]:
    """What the debate established, and what it could not.

    The turn records carry both; until this runs, neither reaches the memo.
    Aggregated here rather than in a state channel because the debate nodes
    are cyclic — a plain list channel would have each turn clobber the last.

    The round_cap gap is the one that earns its keep: a capped debate reads
    in the memo exactly like a resolved one unless the memo says otherwise,
    the same failure `truncated_by_cap` exists to prevent for the news digest.
    """
    turns = state.get("debate_turns") or []
    gaps: list[str] = []
    evidence: list[str] = []

    if not turns:
        reason = state.get("debate_terminated_by") or "unknown"
        gaps.append(
            f"no debate took place ({reason}) — this memo carries no adversarial "
            f"review of the analyst findings, which is not the same as those "
            f"findings having survived one"
        )
        return gaps, evidence

    # Counted across turns but listed once each. A figure the debate leans on
    # gets restated every round, and on the first live run that filled the
    # whole display budget with "0.15, 0.15, 0.15" while two other flagged
    # figures were hidden behind "(+1 more)" — the repetition told the reader
    # nothing and cost them the part that would have.
    flagged = Counter(f for t in turns for f in t.guard_flags)
    if flagged:
        shown = [
            f"{fig} (x{n})" if n > 1 else fig
            for fig, n in flagged.most_common(5)
        ]
        gaps.append(
            f"{sum(flagged.values())} mention(s) of {len(flagged)} figure(s) in the "
            f"debate did not appear in any analyst report and may be fabricated: "
            f"{', '.join(shown)}"
            + (f" (+{len(flagged) - 5} more)" if len(flagged) > 5 else "")
        )

    unquoted = sorted({c for t in turns for c in t.unquoted_evidence})
    if unquoted:
        gaps.append(
            f"{len(unquoted)} claim(s) cite a report but the quoted span is not in "
            f"it: {', '.join(unquoted[:5])}"
            + (f" (+{len(unquoted) - 5} more)" if len(unquoted) > 5 else "")
        )

    # A claim_id is meant to name one stable assertion, reused across turns.
    # Nothing stops a later occurrence from carrying different text — found
    # live (ACN, 2026-08-24): one id named two different claims across two
    # turns. Aggregation by claim_id (Phase 6's risk debate is the reason
    # this matters) must read through `canonical_claims`, not this list —
    # the caveat only makes the drift visible, it does not resolve it.
    drifted = sorted({cid for t in turns for cid in t.claim_text_drift})
    if drifted:
        gaps.append(
            f"{len(drifted)} claim_id(s) were reused with different wording across "
            f"turns, so they do not name one stable assertion: {', '.join(drifted[:5])}"
            + (f" (+{len(drifted) - 5} more)" if len(drifted) > 5 else "")
        )

    if state.get("debate_terminated_by") == "round_cap":
        gaps.append(
            f"the debate hit the {len(turns) // 2}-round cap rather than resolving — "
            f"both sides still had new claims when it stopped, so the transcript is "
            f"a truncated argument, not a concluded one"
        )

    concessions = [t for t in turns if t.payload.stance == "concede"]
    evidence.append(
        f"{len(turns)}-turn bull/bear debate over the analyst reports; "
        f"{len(concessions)} structurally-justified concession(s)"
    )
    return gaps, evidence


def _risk_caveats(state: TradingState) -> tuple[list[str], list[str], list]:
    """What the risk panel established, and what it could not — same role
    as `_debate_caveats`, one cycle up. Also returns the ledger itself,
    since synthesis needs it to resolve `[RF00]`-style citations and
    `run_synthesis` should not have to rebuild it a second time from
    risk_turns.
    """
    turns = state.get("risk_turns") or []
    gaps: list[str] = []
    evidence: list[str] = []

    if not turns:
        reason = state.get("risk_terminated_by") or "unknown"
        gaps.append(
            f"no risk panel ran ({reason}) — this memo carries no dedicated "
            f"risk-factor ledger, which is not the same as the position "
            f"having no risks"
        )
        return gaps, evidence, []

    ledger = build_risk_ledger(turns)
    contested = [e for e in ledger if e.contested]

    flagged = Counter(f for t in turns for f in t.guard_flags)
    if flagged:
        shown = [f"{fig} (x{n})" if n > 1 else fig for fig, n in flagged.most_common(5)]
        gaps.append(
            f"{sum(flagged.values())} guard flag(s) across {len(flagged)} distinct "
            f"issue(s) in the risk panel: {', '.join(shown)}"
            + (f" (+{len(flagged) - 5} more)" if len(flagged) > 5 else "")
        )

    unquoted = sorted({c for t in turns for c in t.unquoted_evidence})
    if unquoted:
        gaps.append(
            f"{len(unquoted)} risk factor(s) cite a report or the debate but the "
            f"quoted span is not in it: {', '.join(unquoted[:5])}"
            + (f" (+{len(unquoted) - 5} more)" if len(unquoted) > 5 else "")
        )

    missing_any = [e for e in ledger if e.missing_scores]
    if missing_any:
        gaps.append(
            f"{len(missing_any)} of {len(ledger)} risk factor(s) are missing a "
            f"score from at least one panelist — a gap in the ledger, not a "
            f"neutral score"
        )

    if state.get("risk_terminated_by") == "round_cap" and contested:
        gaps.append(
            f"{len(contested)} risk factor(s) remained contested (panelists "
            f"diverged by 2+ on severity or likelihood) when the panel hit its "
            f"round cap — the ledger did not converge on these"
        )

    evidence.append(
        f"{len(turns)}-turn three-persona risk panel produced a {len(ledger)}-"
        f"factor ledger; {len(contested)} contested"
    )
    return gaps, evidence, ledger


async def synthesizer_node(state: TradingState) -> dict:
    print(f"[synthesizer] running for {state['ticker']}")
    as_of = state.get("as_of_date")
    if as_of is None:
        # Same rule as news_node, for the same reason: date.today() belongs at
        # the CLI boundary and nowhere else. Defaulting here would silently
        # misdate every historical probe — the memo would claim to be current
        # as of today while describing a window from months earlier.
        raise ValueError(
            "as_of_date missing from TradingState — refusing to date a memo by "
            "wall-clock time. The memo's data_as_of_date must be the run's "
            "analysis date, not whenever the synthesizer happened to execute."
        )
    missing = sorted(
        name for name, key in ANALYST_OUTPUTS.items() if state.get(key) is None
    )
    news_gaps, news_evidence = _news_caveats(state)
    debate_gaps, debate_evidence = _debate_caveats(state)
    risk_gaps, risk_evidence, ledger = _risk_caveats(state)

    base_gaps = (
        [
            f"{name} analyst did not run — this memo carries no {name} evidence "
            f"at all, which is not the same as that evidence being neutral"
            for name in missing
        ]
        + news_gaps
        + debate_gaps
        + risk_gaps
    )
    base_evidence = news_evidence + debate_evidence + risk_evidence

    memo = await run_synthesis(
        state, ledger=ledger, base_gaps=base_gaps, base_evidence=base_evidence, as_of=as_of
    )
    return {"decision_memo": memo}