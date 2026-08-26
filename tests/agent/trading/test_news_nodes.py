"""news_node wiring: the as_of guard, the empty-window case (a valid
result, not an error), and the lookahead post-assertion.

Network and LLM are monkeypatched at the nodes-module seam — the same
names news_node actually calls — so these tests exercise the real node
body, not a reimplementation of it.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

import app.agent.trading.application.nodes as nodes
from app.agent.trading.domain.decision_memo import DecisionMemo, Verdict
from app.agent.trading.domain.news_digest import NewsDigest, NewsItem, SentimentSummary

AS_OF = date(2025, 3, 15)


@pytest.fixture(autouse=True)
def _stub_synthesis(monkeypatch):
    """This file is about what the NEWS leg contributes to the memo's
    data_gaps/evidence via `_news_caveats` — not about the synthesis LLM
    call, which every test below would otherwise hit for real through
    `nodes.synthesizer_node`. Stubbed at the same seam
    test_debate_graph.py's `_stub_synthesis` uses: the memo is built
    directly from the caveats synthesizer_node already computed in Python
    (base_gaps/base_evidence), so every assertion on memo.data_gaps /
    memo.evidence below still exercises the real caveat logic."""
    async def fake_run_synthesis(state, *, ledger, base_gaps, base_evidence, as_of, client=None):
        return DecisionMemo(
            ticker=state["ticker"],
            bull_case="stub bull",
            bear_case="stub bear",
            research_thesis="stub thesis",
            risk_debate_summary="stub risk narrative",
            technical_signal="NOT RUN — technical analyst was excluded from this run",
            reasoning="stub reasoning",
            watch_items=[],
            verdict=Verdict.HOLD,
            confidence=0.0,
            data_as_of_date=as_of,
            data_gaps=base_gaps,
            assumptions=[],
            evidence=base_evidence,
        )

    monkeypatch.setattr(nodes, "run_synthesis", fake_run_synthesis)


def _item(pub: date, sentiment: str = "neutral", relevance: str = "primary") -> NewsItem:
    return NewsItem(
        headline=f"story on {pub}",
        published_date=pub,
        source="wire",
        url="https://example.com/x",
        summary="one line",
        sentiment=sentiment,
        relevance=relevance,
    )


def _patch_ports(monkeypatch, raw, items, issues=None, cost=0.01):
    async def fake_fetch(ticker, as_of, lookback_days=14):
        return raw, as_of - timedelta(days=lookback_days)

    async def fake_digest(articles, ticker):
        return items, issues or [], cost

    monkeypatch.setattr(nodes, "fetch_company_news", fake_fetch)
    monkeypatch.setattr(nodes, "build_digest", fake_digest)


@pytest.mark.anyio
async def test_news_node_refuses_to_run_without_as_of_date():
    with pytest.raises(ValueError, match="as_of_date missing"):
        await nodes.news_node({"ticker": "ACN"})


# ---------------------------------------------------------------------------
# Test 5 — empty result is a valid result
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_quiet_ticker_yields_empty_digest_without_error(monkeypatch):
    _patch_ports(monkeypatch, raw=[], items=[], cost=None)

    update = await nodes.news_node({"ticker": "ACN", "as_of_date": AS_OF})

    digest = update["news_digest"]
    assert isinstance(digest, NewsDigest)
    assert digest.items == []
    assert digest.raw_article_count == 0
    assert digest.as_of_date == AS_OF
    assert update["news_digest_issues"] == []


@pytest.mark.anyio
async def test_news_node_builds_digest_with_provenance_counts(monkeypatch):
    raw = [
        {"headline": "kept", "datetime": 1741780800, "source": "wire",
         "url": "https://example.com/1", "summary": "b"},
        {"headline": "late", "datetime": 1750000000, "source": "wire",
         "url": "https://example.com/2", "summary": "b"},
        {"headline": "no ts", "datetime": 0, "source": "wire",
         "url": "https://example.com/3", "summary": "b"},
    ]
    _patch_ports(monkeypatch, raw=raw, items=[_item(date(2025, 3, 12), "positive")],
                 issues=["missing index 9"])

    update = await nodes.news_node({"ticker": "ACN", "as_of_date": AS_OF})

    digest = update["news_digest"]
    assert digest.raw_article_count == 3
    # the real filter_and_dedup ran: one in-window, one late, one zero-ts
    assert digest.deduped_count == 1
    assert digest.dropped_out_of_window == 1
    assert digest.dropped_missing_date == 1
    assert digest.truncated_by_cap is False
    assert update["news_digest_issues"] == ["missing index 9"]


# ---------------------------------------------------------------------------
# Lookahead post-assertion
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_post_dated_item_in_digest_raises_lookahead_leak(monkeypatch):
    _patch_ports(
        monkeypatch,
        raw=[],
        items=[_item(AS_OF), _item(AS_OF + timedelta(days=2))],
    )

    with pytest.raises(AssertionError, match="Lookahead leak"):
        await nodes.news_node({"ticker": "ACN", "as_of_date": AS_OF})


# ---------------------------------------------------------------------------
# sentiment_node — deterministic aggregation, no LLM
# ---------------------------------------------------------------------------

def _digest(items: list[NewsItem]) -> NewsDigest:
    return NewsDigest(
        ticker="ACN",
        as_of_date=AS_OF,
        window_start=AS_OF - timedelta(days=14),
        items=items,
        raw_article_count=len(items),
        deduped_count=len(items),
        dropped_out_of_window=0,
        dropped_missing_date=0,
        truncated_by_cap=False,
    )


@pytest.mark.anyio
async def test_sentiment_node_aggregates_counts_and_net_score():
    items = [
        _item(AS_OF, "positive"),
        _item(AS_OF - timedelta(days=1), "positive"),
        _item(AS_OF - timedelta(days=2), "positive"),
        _item(AS_OF - timedelta(days=3), "negative"),
        _item(AS_OF - timedelta(days=4), "neutral"),
    ]

    update = await nodes.sentiment_node({"news_digest": _digest(items)})

    s = update["sentiment_summary"]
    assert (s.positive, s.negative, s.neutral) == (3, 1, 1)
    assert s.article_count == 5
    assert s.excluded_by_relevance == 0
    assert s.net_score == pytest.approx((3 - 1) / 5)
    assert s.ticker == "ACN" and s.as_of_date == AS_OF


@pytest.mark.anyio
async def test_sentiment_node_counts_only_primary_relevance():
    """The fix for the measured failure: a vendor feed tagged with this
    ticker but dominated by sector coverage must not have that coverage
    counted as sentiment about this company. Here every excluded article is
    negative, so including them would flip the sign of net_score."""
    items = [
        _item(AS_OF, "positive", "primary"),
        _item(AS_OF - timedelta(days=1), "positive", "primary"),
        _item(AS_OF - timedelta(days=2), "negative", "mentioned"),
        _item(AS_OF - timedelta(days=3), "negative", "mentioned"),
        _item(AS_OF - timedelta(days=4), "negative", "unrelated"),
        _item(AS_OF - timedelta(days=5), "negative", "unrelated"),
    ]

    update = await nodes.sentiment_node({"news_digest": _digest(items)})

    s = update["sentiment_summary"]
    assert (s.positive, s.negative, s.neutral) == (2, 0, 0)
    assert s.article_count == 2
    assert s.excluded_by_relevance == 4
    assert s.net_score == pytest.approx(1.0)
    # the aggregate covers only what it counted, and says so
    assert s.article_count + s.excluded_by_relevance == len(items)


@pytest.mark.anyio
async def test_stale_checkpoint_items_fail_loudly_not_as_attribute_error():
    """Observed against a real Postgres checkpoint written before `relevance`
    existed: the outer NewsDigest deserializes fine, but its items come back
    as plain dicts because pydantic cannot rebuild a child missing a
    now-required field — and nothing raises at the read. The first symptom
    was `AttributeError: 'dict' object has no attribute 'relevance'`, several
    frames from the cause.

    model_construct reproduces that state, since NewsDigest(...) would
    validate the dicts away.
    """
    degraded = NewsDigest.model_construct(
        ticker="ACN",
        as_of_date=AS_OF,
        window_start=AS_OF - timedelta(days=14),
        items=[{"headline": "old", "sentiment": "positive"}],
        raw_article_count=1,
        deduped_count=1,
        dropped_out_of_window=0,
        dropped_missing_date=0,
        truncated_by_cap=False,
    )

    with pytest.raises(TypeError, match="predates a NewsItem schema change"):
        await nodes.sentiment_node({"news_digest": degraded})


@pytest.mark.anyio
async def test_all_articles_filtered_out_is_distinguishable_from_no_news():
    """Both cases give net_score 0.0 with article_count 0. Only
    excluded_by_relevance separates 'nothing was about this company' from
    'there was no news', and they justify different confidence."""
    noisy = await nodes.sentiment_node(
        {"news_digest": _digest([_item(AS_OF, "positive", "unrelated")] * 5)}
    )
    quiet = await nodes.sentiment_node({"news_digest": _digest([])})

    assert noisy["sentiment_summary"].article_count == 0
    assert noisy["sentiment_summary"].net_score == 0.0
    assert noisy["sentiment_summary"].excluded_by_relevance == 5

    assert quiet["sentiment_summary"].article_count == 0
    assert quiet["sentiment_summary"].net_score == 0.0
    assert quiet["sentiment_summary"].excluded_by_relevance == 0


@pytest.mark.anyio
async def test_sentiment_node_handles_empty_digest_as_valid():
    """Test 5's second half: absence of news is a valid result. net_score
    is 0.0 with article_count 0 — not neutrality-with-evidence."""
    update = await nodes.sentiment_node({"news_digest": _digest([])})

    s = update["sentiment_summary"]
    assert s.article_count == 0
    assert s.net_score == 0.0
    assert s.excluded_by_relevance == 0
    assert (s.positive, s.negative, s.neutral) == (0, 0, 0)


# ---------------------------------------------------------------------------
# What the news leg contributes to the DecisionMemo
# ---------------------------------------------------------------------------

def _synth_state(**over) -> dict:
    digest = _digest([_item(AS_OF, "positive")])
    state = {
        "ticker": "ACN",
        "as_of_date": AS_OF,
        "news_digest": digest,
        "news_digest_issues": [],
        "sentiment_summary": SentimentSummary(
            ticker="ACN", as_of_date=AS_OF, positive=8, negative=2, neutral=2,
            net_score=0.5, article_count=12, excluded_by_relevance=40,
        ),
    }
    state.update(over)
    return state


@pytest.mark.anyio
async def test_memo_is_dated_by_the_run_not_the_wall_clock():
    """date.today() belongs at the CLI boundary and nowhere else. Dating the
    memo by execution time silently misdates every historical probe: the memo
    would claim to be current as of today while describing a window from
    months earlier."""
    probe = date(2025, 3, 1)
    state = _synth_state(as_of_date=probe)
    state["news_digest"] = NewsDigest(
        ticker="ACN", as_of_date=probe, window_start=probe - timedelta(days=14),
        items=[], raw_article_count=0, deduped_count=0, dropped_out_of_window=0,
        dropped_missing_date=0, truncated_by_cap=False,
    )

    memo = (await nodes.synthesizer_node(state))["decision_memo"]

    assert memo.data_as_of_date == probe
    assert memo.data_as_of_date != date.today()


@pytest.mark.anyio
async def test_synthesizer_refuses_to_date_a_memo_without_as_of():
    state = _synth_state()
    del state["as_of_date"]

    with pytest.raises(ValueError, match="refusing to date a memo"):
        await nodes.synthesizer_node(state)


@pytest.mark.anyio
async def test_truncated_digest_is_declared_a_sample_in_the_memo():
    """A truncated digest saw part of the window. A reader acting on the memo
    alone previously had no way to know that — the flag existed on the digest
    and stopped there."""
    state = _synth_state()
    state["news_digest"] = NewsDigest(
        ticker="ACN", as_of_date=AS_OF, window_start=AS_OF - timedelta(days=14),
        items=[_item(AS_OF, "positive")], raw_article_count=500, deduped_count=300,
        dropped_out_of_window=0, dropped_missing_date=0, truncated_by_cap=True,
    )

    memo = (await nodes.synthesizer_node(state))["decision_memo"]
    gaps = " ".join(memo.data_gaps)

    assert "SAMPLE" in gaps and "500" in gaps
    assert "skews to the most recent days" in gaps


@pytest.mark.anyio
async def test_digest_integrity_issues_reach_the_memo():
    state = _synth_state(
        news_digest_issues=["missing index 3: 'ACN cuts guidance'", "duplicate index 1"]
    )

    memo = (await nodes.synthesizer_node(state))["decision_memo"]
    gaps = " ".join(memo.data_gaps)

    assert "2 digest integrity issue(s)" in gaps
    assert "ACN cuts guidance" in gaps


@pytest.mark.anyio
async def test_sentiment_signal_becomes_memo_evidence():
    memo = (await nodes.synthesizer_node(_synth_state()))["decision_memo"]

    assert len(memo.evidence) == 1
    ev = memo.evidence[0]
    assert "+0.50" in ev and "12 article(s)" in ev
    assert "40 further article(s)" in ev      # the excluded count travels too


@pytest.mark.anyio
async def test_zero_relevant_articles_is_declared_absence_not_neutrality():
    """The failure this guards: a 0.00 score read as 'the market is
    indifferent' when it actually means nothing in the window was about the
    company."""
    state = _synth_state(
        sentiment_summary=SentimentSummary(
            ticker="ACN", as_of_date=AS_OF, positive=0, negative=0, neutral=0,
            net_score=0.0, article_count=0, excluded_by_relevance=60,
        )
    )

    memo = (await nodes.synthesizer_node(state))["decision_memo"]
    gaps = " ".join(memo.data_gaps)

    assert "ABSENCE of evidence" in gaps
    assert "60 were excluded" in gaps
    assert memo.evidence == []      # nothing was found, so nothing is claimed


@pytest.mark.anyio
async def test_thin_sample_is_flagged_but_a_healthy_one_is_not():
    thin = _synth_state(
        sentiment_summary=SentimentSummary(
            ticker="ACN", as_of_date=AS_OF, positive=2, negative=0, neutral=0,
            net_score=1.0, article_count=2, excluded_by_relevance=5,
        )
    )

    thin_memo = (await nodes.synthesizer_node(thin))["decision_memo"]
    healthy_memo = (await nodes.synthesizer_node(_synth_state()))["decision_memo"]

    assert any("rests on only 2 article(s)" in g for g in thin_memo.data_gaps)
    assert not any("rests on only" in g for g in healthy_memo.data_gaps)


@pytest.mark.anyio
async def test_a_clean_digest_adds_no_spurious_caveats():
    """The caveats must be earned. A full, healthy digest should leave the
    memo's gap list exactly as it was before the news leg contributed."""
    memo = (await nodes.synthesizer_node(_synth_state()))["decision_memo"]

    for noise in ("SAMPLE", "integrity issue", "ABSENCE", "rests on only"):
        assert not any(noise in g for g in memo.data_gaps), noise


@pytest.mark.anyio
async def test_news_not_run_still_reports_it_and_claims_nothing():
    """A skipped analyst and a quiet one must stay distinguishable."""
    state = _synth_state()
    del state["news_digest"], state["sentiment_summary"]

    memo = (await nodes.synthesizer_node(state))["decision_memo"]

    assert any("news analyst did not run" in g for g in memo.data_gaps)
    assert memo.evidence == []
