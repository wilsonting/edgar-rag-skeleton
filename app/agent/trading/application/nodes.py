"""Application nodes for the trading graph.

fundamentals (Phase 2), technical (Phase 3), news/sentiment (Phase 4), the
debate (Phase 5, in debate_nodes.py) and the risk panel (Phase 6, in
risk_nodes.py) are real. The synthesizer's own LLM call and citation
resolution live in infrastructure/synthesis_port.py — this module still owns
the caveat computation (`_news_caveats`, `_debate_caveats`, `_risk_caveats`),
same split as the risk/debate ports vs. their nodes.
"""
import logging
from collections import Counter
from datetime import date, datetime, timezone

from app.infrastructure.llm import get_client

from app.agent.trading.application.guards import check_run_guards
from app.agent.trading.application.risk_ledger import (
    build_risk_ledger,
    unexpected_missing_scores,
)
from app.agent.trading.application.risk_router import RISK_MAX_TURNS
from app.agent.trading.domain.budget import NodeBudgetExceeded, RunTermination
from app.agent.trading.domain.decision_memo import Verdict
from app.agent.trading.domain.errors import VendorError
from app.agent.trading.domain.news_digest import (
    AGGREGATED_RELEVANCE,
    NewsDigest,
    NewsItem,
    SentimentSummary,
)
from app.agent.trading.domain.risk import PERSONAS
from app.agent.trading.domain.technical_report import TechnicalReport
from app.agent.trading.domain.trading_state import TradingState
from app.agent.trading.infrastructure.fundamentals_port import get_fundamentals_report
from app.agent.trading.infrastructure.news_data_port import fetch_company_news, filter_and_dedup
from app.agent.trading.infrastructure.news_digest_port import build_digest
from app.agent.trading.infrastructure.price_data_port import get_price_history
from app.agent.trading.application.technical_indicators import compute_indicators
from app.agent.trading.infrastructure.synthesis_port import (
    MemoVerificationError,
    SynthesisFabricationError,
    SynthesisReferenceError,
    run_synthesis,
    verdict_agreement,
    verify_decision_memo,
)
from app.agent.trading.infrastructure.decision_memo_port import (
    save_aborted_run_memo,
    save_failed_decision_memo,
)
from app.agent.trading.infrastructure.technical_interpreter_port import interpret_indicators, save_technical_report

# Phase 6 exit-criteria fix (2026-08-26, code review): post-fix measurement
# on two tickers (AVGO, ASML) showed the risk panel's verdict genuinely
# splits direction across independent samples of the SAME fixed debate —
# not an identity artifact (slate-identity and threshold-brittleness fixes
# both landed first; the split persisted). A fixed-ledger repeat of the
# Risk Judge alone (3 calls against one frozen ledger, AVGO) came back
# unanimous, which localizes the variance to the PANEL, not the Judge — so
# sampling has to re-run the whole (panel, Research Manager, Risk Judge)
# trial, not just resample the Judge's call. See
# trading-agent-known-gaps.md for the measurements this decision rests on.
logger = logging.getLogger(__name__)

RISK_VERDICT_SAMPLES = 3


async def graceful_abort_node(state: TradingState) -> dict:
    """The sole destination of every guard edge's "abort" branch (graph.py's
    `guarded()`) — a budget or deadline breach, never a debate/risk-quality
    outcome (see application/guards.py's module docstring for why those stay
    separate). Recomputes the reason rather than having the router pass it
    along, same pattern debate_close_node uses for `debate_terminated_by`:
    a router returns only a routing decision, and whatever a node needs
    beyond that it re-derives from state.

    Writes a partial artifact and a cost-log run_summary line so an aborted
    run is never silently indistinguishable from one that never ran —
    same "a truncated result must say so" principle behind
    debate_terminated_by/risk_terminated_by, one level up.
    """
    events = state.get("cost_events") or []
    if state.get("node_budget_breach"):
        # A port's own cap tripped (graph.py's _contain_node_budget) — the
        # run-level budget may well be intact, so it is not re-derived here.
        terminated_by = RunTermination.NODE_BUDGET_EXCEEDED
    else:
        terminated_by = check_run_guards(events, state["budget"], datetime.now(timezone.utc))
    if terminated_by is None:
        # Only reachable if a guard edge routed here without the guard
        # actually tripping — a wiring bug, named here rather than silently
        # writing an artifact that claims a breach that didn't happen.
        raise RuntimeError(
            "graceful_abort_node entered but check_run_guards found no "
            "breach — a guard edge routed here incorrectly."
        )
    print(f"[abort] run terminated: {terminated_by.value}")
    vault_path = save_aborted_run_memo(state, terminated_by)
    print(f"[abort] saved partial artifact to {vault_path}")
    return {"run_terminated_by": terminated_by}


async def fundamentals_node(state: TradingState) -> dict:
    as_of = state.get("as_of_date")
    if as_of is None:
        # Same rule as technical_node and news_node, and the last leg to
        # adopt it. This node used to call date.today() inside the port and
        # never see the run's analysis date at all, so a historical run
        # bounded its prices and news and let its most heavily-weighted
        # analyst read whatever had been filed since.
        raise ValueError(
            "as_of_date missing from TradingState — refusing to run the "
            "fundamentals agent unbounded. Filing retrieval without an "
            "explicit upper bound is a lookahead bug."
        )
    print(f"[fundamentals] running for {state['ticker']} as of {as_of}")
    # The budget and the run's earlier spend go in so the agent loop can stop
    # itself: the edge guard around this node cannot see inside it.
    report = await get_fundamentals_report(
        state["ticker"],
        as_of,
        run_id=state.get("run_id"),
        budget=state.get("budget"),
        prior_events=state.get("cost_events") or [],
    )
    # cost_event is None on a cache hit — nothing was spent this run. report
    # itself can be None too (a test double simulating "analyst did not
    # run"; the real port never returns None).
    # Both the agent's own loop AND what its tools spent server-side. The
    # second used to be invisible to this ledger, which is what let a run
    # exceed its budget without `check_run_guards` ever seeing it.
    events = []
    if report:
        events = [e for e in (report.cost_event, *report.tool_cost_events) if e]
    return {"fundamentals_report": report, "cost_events": events}


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

    try:
        df, source, dropped_bars = await get_price_history(ticker, as_of)
    except VendorError as exc:
        # A vendor outage ended the whole run, discarding the fundamentals
        # leg that had already completed and been paid for — $0.070 on FIG,
        # 2026-09-13, when yfinance returned an empty frame and Finnhub's
        # free tier 403'd on historical candles. Nothing about a price feed
        # being down invalidates the filing analysis.
        #
        # Degrading here is not silence: the synthesizer already reports an
        # absent analyst as a data gap, and `analyst_failures` makes this
        # one say WHY, which an unselected analyst cannot claim. The one
        # thing that must not happen is a memo that reads as though the
        # technical evidence were neutral.
        print(f"[technical] FAILED, continuing without it: {exc}")
        logger.warning("technical analyst failed for %s: %s", ticker, exc)
        return {"analyst_failures": [f"technical: {exc}"]}
    if dropped_bars:
        print(f"[technical] dropped {dropped_bars} incomplete bar(s) from {source}")
    indicators = compute_indicators(df)
    interpretation, flagged, flagged_claims, cost_usd, cost_event = await interpret_indicators(
        ticker, indicators, run_id=state.get("run_id")
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
        cost_event=cost_event,
    )
    vault_path = save_technical_report(report, cost_usd=cost_usd)
    print(f"[technical] saved report to {vault_path}")
    return {"technical_report": report, "cost_events": [cost_event]}


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

    items, issues, cost_usd, cost_event, sanitizer_flags = await build_digest(
        clean, ticker, run_id=state.get("run_id")
    )

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
        cost_event=cost_event,
        sanitizer_flags=sanitizer_flags,
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
    if sanitizer_flags:
        print(f"[news] sanitizer flagged {len(sanitizer_flags)} article(s): {sanitizer_flags}")
    events = [cost_event] if cost_event else []
    return {"news_digest": digest, "news_digest_issues": issues, "cost_events": events}


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


def _missing_analyst_gaps(missing: list[str], failures: list[str]) -> list[str]:
    """One gap per analyst with no report, saying WHICH kind of absence.

    An analyst that was never selected (`--only`) and one whose vendor was
    down leave the same hole, but they are different claims about it: the
    first means nobody asked, the second means somebody asked and the answer
    could not be got. A memo that renders a vendor outage as "did not run"
    invites the reader to treat the silence as deliberate.

    Neither ever reads as neutral evidence — that is the point of saying
    anything at all.
    """
    reasons = {
        name.strip(): reason.strip()
        for name, _, reason in (f.partition(":") for f in failures)
        if reason.strip()
    }
    gaps = []
    for name in missing:
        if name in reasons:
            gaps.append(
                f"{name} analyst FAILED and this memo carries no {name} "
                f"evidence as a result: {reasons[name]}"
            )
        else:
            gaps.append(
                f"{name} analyst did not run — this memo carries no {name} "
                f"evidence at all, which is not the same as that evidence "
                f"being neutral"
            )
    return gaps


def _fundamentals_caveats(state: TradingState) -> list[str]:
    """What the fundamentals leg could not see, for a run dated in the past.

    Filing retrieval is bounded at `as_of_date` (fundamentals_port), so a
    historical run reads only what had been filed by then — which is the
    point. What the bound cannot reach is the model's own priors: it may
    know perfectly well how the year turned out. Said plainly in the memo
    rather than assumed, because "bounded retrieval" and "no knowledge of
    later events" are not the same claim, and only the first is enforced.

    Nothing to say on a run dated today: there is no "after" to leak.
    """
    as_of = state.get("as_of_date")
    report = state.get("fundamentals_report")
    if as_of is None or report is None or as_of >= date.today():
        return []
    return [
        f"this is a historical run dated {as_of.isoformat()}: filing "
        f"retrieval was bounded at that date, so the fundamentals leg read "
        f"nothing filed after it — but the model's own prior knowledge is "
        f"not bounded, and may include how the period turned out"
    ]


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

    unresolved = [u for t in turns for u in t.unresolved_flags]
    if unresolved:
        gaps.append(
            f"{len(unresolved)} turn(s) in the debate pointed at something not in "
            f"the transcript: {', '.join(unresolved[:5])}"
            + (f" (+{len(unresolved) - 5} more)" if len(unresolved) > 5 else "")
        )

    # Listed in full rather than counted: "3 sentence(s) contradicted their
    # figures" tells a reader nothing they can act on, and the finding is
    # short enough to print.
    directions = [d for t in turns for d in t.direction_flags]
    if directions:
        gaps.append(
            f"{len(directions)} sentence(s) in the debate state a direction the "
            f"cited figures contradict: {'; '.join(directions[:3])}"
            + (f" (+{len(directions) - 3} more)" if len(directions) > 3 else "")
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

    # Counts BOTH shapes — a full concession (stance='concede') and a partial
    # one (an opposing claim_id named in `concession_trigger` on a hold or
    # sharpen turn). Counting only the first is what made this line read
    # "0 structurally-justified concession(s)" for every debate the pipeline
    # has ever run: see check_concession's docstring for why no turn was ever
    # stance='concede'.
    concessions = [
        t for t in turns
        if t.payload.stance == "concede" or t.payload.concession_trigger
    ]
    evidence.append(
        f"{len(turns)}-turn bull/bear debate over the analyst reports; "
        f"{len(concessions)} structurally-justified concession(s)"
    )
    # Zero is reported as a gap, not as a finding. It means no turn NAMED an
    # opposing claim_id, which is not the same as neither side moving — a
    # concession written only in the argument prose is invisible here, and the
    # memo must not read this count as "the debate was unmoved".
    if turns and not concessions:
        gaps.append(
            "no turn named an opposing claim_id in `concession_trigger`, so no "
            "concession is recorded structurally — this is not evidence that "
            "neither side moved, since a concession made only in the argument "
            "prose is not counted"
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

    # Only the absences nobody asked for — see `unexpected_missing_scores`.
    # Counting every gap in `missing_scores` reported the neutral's
    # protocol-required silence on uncontested factors as a defect, in every
    # run of the 2026-08-29 battery.
    missing_any = [e for e in ledger if unexpected_missing_scores(e)]
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


async def _sample_additional_risk_panel(state: TradingState) -> tuple[list, list]:
    """One fresh 9-turn risk panel over the SAME fixed debate/technical
    context already in `state`, independent of `state["risk_turns"]` — a
    second (or third) vote for `synthesizer_node`'s majority-of-N sampling,
    not a resume of the graph-checkpointed panel. Drives `risk_nodes.
    _risk_turn` directly (same validation as a real graph turn: rotation,
    ordering, stale-checkpoint checks) rather than calling
    `risk_port.run_risk_turn` here directly, so this goes through the same
    module attribute tests already monkeypatch
    (`risk_nodes.run_risk_turn`) — calling the port function straight from
    this module would silently bypass that seam and hit the network in
    every existing synthesizer_node test that populates `risk_turns`.
    Deliberately NOT checkpointed per-turn like the graph's own panel is —
    these extra samples live and die inside this one synthesizer node call,
    same resumability granularity `run_synthesis` already had before this
    change."""
    # Local import: risk_nodes -> risk_port -> debate_port -> nodes (for
    # ANALYST_OUTPUTS) is a real cycle at module-load time — same reason
    # every debate_port import inside synthesis_port.py is function-local.
    import app.agent.trading.application.risk_nodes as risk_nodes

    turns: list = []
    cost_events: list = []
    for i in range(RISK_MAX_TURNS):
        persona = PERSONAS[i % len(PERSONAS)]
        result = await risk_nodes._risk_turn({**state, "risk_turns": turns}, persona)
        turns.append(result["risk_turns"][0])
        # Real spend regardless of which sample wins the vote — see the
        # identical reasoning in synthesizer_node's own loop for why this
        # must not be dropped just because it's a "sampling" turn.
        cost_events.extend(result.get("cost_events") or [])
    return turns, cost_events


def _verify_or_raise(memo, state, ledger, debate_turns):
    """Phase 7: independent re-check of the memo synthesizer_node is about
    to return, distinct from the per-call guards that already ran during
    generation (SynthesisFabricationError/SynthesisReferenceError catch a
    bad call before its output is used; this catches a bad ASSEMBLED
    artifact). A failure here means a bug slipped past guards that each
    looked clean on their own — fail loud, but write the memo first (same
    pattern technical_node/fundamentals_node already use to save from
    inside a still-executing graph): the failed memo is the debugging
    artifact, and raising without saving it would re-run the pipeline
    blind on the next attempt."""
    result = verify_decision_memo(memo, state, ledger, debate_turns)
    if not result.passed:
        vault_path = save_failed_decision_memo(memo, result)
        raise MemoVerificationError(
            f"the assembled memo for {state['ticker']} failed post-hoc "
            f"verification — unbacked number(s): {result.unbacked_numbers}, "
            f"unresolved reference(s): {result.unresolved_references}. Every "
            f"per-call guard passed, so this is an assembly-step bug, not an "
            f"ordinary fabrication. Failed memo saved to {vault_path} for "
            f"debugging."
        )
    return result


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
    fundamentals_gaps = _fundamentals_caveats(state)
    news_gaps, news_evidence = _news_caveats(state)
    debate_gaps, debate_evidence = _debate_caveats(state)
    risk_gaps, risk_evidence, ledger = _risk_caveats(state)

    base_gaps = (
        _missing_analyst_gaps(missing, state.get("analyst_failures") or [])
        + fundamentals_gaps
        + news_gaps
        + debate_gaps
        + risk_gaps
    )
    base_evidence = news_evidence + debate_evidence + risk_evidence

    debate_turns = state.get("debate_turns") or []

    if not ledger:
        # No risk panel ran (e.g. `--only technical`) — nothing to sample,
        # same single-call behavior as before this change. `verdict_samples`
        # stays its default empty list, which is the honest signal that
        # sampling did not run, not that it ran and produced one entry.
        memo = await run_synthesis(
            state, ledger=ledger, base_gaps=base_gaps, base_evidence=base_evidence, as_of=as_of
        )
        _verify_or_raise(memo, state, ledger, debate_turns)
        return {"decision_memo": memo, "cost_events": memo.cost_events}

    # Each of the RISK_VERDICT_SAMPLES trials is an independent (panel,
    # Research Manager, Risk Judge) run, and the Research Manager/Risk Judge
    # calls carry a citation/fabrication guard that raises on an untrusted
    # output (SynthesisFabricationError, SynthesisReferenceError). Measured
    # live hit rate is ~1-in-8 calls (trading-agent-known-gaps.md, FIG,
    # 2026-08-26) — at that rate a 3-sample run has better than even odds of
    # tripping it somewhere, and letting one trial's guard crash the whole
    # node threw away every OTHER trial already paid for, including ones
    # that passed cleanly. A trial the guard blocks is dropped from the
    # vote instead of aborting the run; only if EVERY trial is dropped does
    # this raise, since at that point there is genuinely no memo to return.
    memos: list = []
    # Parallel to `memos`: the exact (state, ledger) each entry was
    # generated from. Under majority-of-N sampling every trial ran its own
    # independent risk panel with its own factor ids, so verifying the
    # chosen memo against a DIFFERENT trial's ledger would misreport real
    # citations as unresolved — the context has to travel with its memo.
    contexts: list[tuple[dict, list]] = []
    dropped: list[str] = []
    # Samples that failed with any other error — see the except below.
    errored: list[str] = []
    # Every trial's cost is real regardless of whether its memo survives to
    # the vote — a dropped trial still spent real tokens (see
    # SynthesisFabricationError/SynthesisReferenceError's cost_events, which
    # carry a raising trial's cost out past the except below). Collected
    # here rather than read off `memos` because `memos` only holds SURVIVING
    # trials, and undercounting a run's real spend is exactly the failure
    # mode the run-level budget guard exists to prevent.
    all_cost_events: list = []
    # Routing, not bound to one model: this object is handed to
    # run_synthesis, which uses it for BOTH the Research Manager and the
    # Risk Judge, and those read two different model env vars.
    client = get_client()
    # Checked before each EXTRA sample. Samples 2 and 3 each run a fresh
    # 9-turn risk panel plus the Research Manager and Risk Judge, all inside
    # this one node, where the edge guard cannot see them. The first sample
    # needs no check: the guard on the edge into this node just ran.
    budget = state.get("budget")
    budget_stop = None
    for i in range(RISK_VERDICT_SAMPLES):
        if i > 0 and budget is not None:
            budget_stop = check_run_guards(
                [*(state.get("cost_events") or []), *all_cost_events],
                budget,
                datetime.now(timezone.utc),
            )
            if budget_stop is not None:
                print(
                    f"[synthesizer] {budget_stop.value} before sample {i + 1}/"
                    f"{RISK_VERDICT_SAMPLES} — skipping the remaining sample(s)"
                )
                skipped = RISK_VERDICT_SAMPLES - i
                break
        try:
            if i == 0:
                # The first trial reuses the graph-checkpointed panel already
                # in `state`/`ledger` rather than sampling a fresh one — same
                # as before this change.
                sample_ledger, sample_state, sample_client = ledger, state, None
            else:
                sample_turns, sample_cost_events = await _sample_additional_risk_panel(state)
                all_cost_events.extend(sample_cost_events)
                sample_ledger = build_risk_ledger(sample_turns)
                sample_state = {**state, "risk_turns": sample_turns}
                sample_client = client
            memo = await run_synthesis(
                sample_state, ledger=sample_ledger, base_gaps=base_gaps,
                base_evidence=base_evidence, as_of=as_of, client=sample_client,
            )
        except NodeBudgetExceeded:
            # A node's own cap is a run-level stop, not one bad sample.
            raise
        except (SynthesisFabricationError, SynthesisReferenceError) as exc:
            print(
                f"[synthesizer] sample {i + 1}/{RISK_VERDICT_SAMPLES} dropped by "
                f"the citation/fabrication guard: {exc}"
            )
            dropped.append(str(exc))
            all_cost_events.extend(exc.cost_events)
            continue
        except Exception as exc:
            # Anything else — a second schema failure, a provider error, a
            # risk turn that raised inside an extra panel — used to escape the
            # node and discard every sample already paid for, including ones
            # that had passed. It is one failed vote. Its spend is on disk
            # (log_cost writes first) and in the run summary via the disk
            # reconciliation, though not in this node's cost_events.
            print(
                f"[synthesizer] sample {i + 1}/{RISK_VERDICT_SAMPLES} failed and "
                f"was dropped: {type(exc).__name__}: {exc}"
            )
            errored.append(f"{type(exc).__name__}: {exc}")
            continue
        memos.append(memo)
        contexts.append((sample_state, sample_ledger))
        all_cost_events.extend(memo.cost_events)

    if not memos:
        # Every trial dropped means the whole node raises, so nothing is
        # returned to state here — this trial batch's cost_events (already
        # collected above) are lost from the run-level ledger along with
        # everything else this node would have returned. Narrow: it needs
        # EVERY one of RISK_VERDICT_SAMPLES trials to trip the guard, at a
        # measured ~1-in-8 per-call rate, and the run already fails outright
        # in this case regardless of Phase 8.
        if errored:
            raise RuntimeError(
                f"no risk-verdict sample for {state['ticker']} survived — "
                f"{len(dropped)} dropped by the citation/fabrication guard, "
                f"{len(errored)} failed with an error"
                + (f", the rest skipped on {budget_stop.value}" if budget_stop else "")
                + f": {dropped + errored}"
            )
        if budget_stop is not None:
            raise SynthesisFabricationError(
                f"no risk-verdict sample for {state['ticker']} survived — "
                f"{len(dropped)} dropped by the citation/fabrication guard, the "
                f"rest skipped on {budget_stop.value} — no memo produced: {dropped}"
            )
        raise SynthesisFabricationError(
            f"all {RISK_VERDICT_SAMPLES} risk-verdict samples for {state['ticker']} "
            f"were dropped by the citation/fabrication guard — no memo produced: {dropped}"
        )

    verdicts = [m.verdict.value for m in memos]
    top_verdict, top_count = Counter(verdicts).most_common(1)[0]
    has_majority = top_count > len(memos) / 2

    if has_majority:
        # Reuse the first sample whose OWN verdict agrees with the
        # majority, so the memo's narrative and its verdict label are
        # never inconsistent with each other (a memo arguing `sell` should
        # never be labeled `hold` because sample 1 happened to say `hold`
        # while 2 and 3 said `sell`).
        final_idx = next(i for i, m in enumerate(memos) if m.verdict.value == top_verdict)
        split_note = f"risk verdict sampled N={len(memos)}: {verdicts} — majority {top_verdict}"
    else:
        final_idx = 0
        split_note = (
            f"risk verdict sampled N={len(memos)}: {verdicts} — no majority, "
            f"reported as UNRESOLVED rather than picking one sample's answer"
        )

    final = memos[final_idx]
    final_state, final_ledger = contexts[final_idx]
    if not has_majority:
        final = final.model_copy(update={"verdict": Verdict.UNRESOLVED})

    extra_gaps = [split_note]
    if dropped:
        extra_gaps.append(
            f"{len(dropped)} of {RISK_VERDICT_SAMPLES} risk-verdict sample(s) were "
            f"dropped by the citation/fabrication guard before voting (untrustworthy "
            f"output, not counted) — this verdict reflects only the surviving "
            f"{len(memos)} sample(s), a weaker signal than a full {RISK_VERDICT_SAMPLES}-way vote"
        )
    if errored:
        extra_gaps.append(
            f"{len(errored)} of {RISK_VERDICT_SAMPLES} risk-verdict sample(s) failed "
            f"with an error and were dropped before voting — this verdict reflects "
            f"only the surviving {len(memos)} sample(s): {'; '.join(errored)[:300]}"
        )
    if budget_stop is not None:
        extra_gaps.append(
            f"{skipped} of {RISK_VERDICT_SAMPLES} risk-verdict sample(s) were skipped "
            f"because the run's {budget_stop.value.replace('_', ' ')} was reached — "
            f"this verdict reflects only {len(memos)} sample(s), a weaker signal "
            f"than a full {RISK_VERDICT_SAMPLES}-way vote"
        )

    final = final.model_copy(update={
        "verdict_samples": verdicts,
        # Reported beside `evidence_quality`, never folded into it: one is
        # agreement about the VERDICT across trials, the other is coverage
        # and within-trial factor agreement. Conflating them is what made a
        # 2-1 split able to report 0.97. See `verdict_agreement`.
        "verdict_agreement": verdict_agreement(verdicts),
        "data_gaps": final.data_gaps + extra_gaps,
    })

    # Verify the memo actually being returned — i.e. after every model_copy
    # above, not an intermediate — against the SAME trial it came from.
    # verify_decision_memo itself excludes data_gaps/evidence from its scan
    # (see its docstring), so this ordering has no effect on WHAT gets
    # checked, only the hygiene of checking the actual final object rather
    # than a pre-assembly one.
    verification = _verify_or_raise(final, final_state, final_ledger, debate_turns)

    # Surfaced as a gap rather than a failure, and only on a memo that
    # already passed. These figures have an antecedent — the debate or the
    # risk panel stated them — but no ANALYST REPORT contains them, so
    # their only source is a model. Most are sound derivations from
    # grounded endpoints (a growth rate, a difference); some are not, and
    # containment cannot tell those apart. Gating would fail nearly every
    # memo that states a growth rate, and staying silent is what let the
    # distinction go unnoticed until Phase 9 went looking for it.
    if verification.debate_originated_numbers:
        final = final.model_copy(update={"data_gaps": final.data_gaps + [
            f"{len(verification.debate_originated_numbers)} figure(s) in the "
            f"memo's load-bearing reasoning appear nowhere in any analyst "
            f"report — the debate or risk panel originated them, so they are "
            f"derivations or assertions, not cited evidence: "
            + ", ".join(verification.debate_originated_numbers)
        ]})

    return {"decision_memo": final, "cost_events": all_cost_events}