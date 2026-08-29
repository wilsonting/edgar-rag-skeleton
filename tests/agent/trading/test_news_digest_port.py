"""Batch-integrity guard for the Haiku digest (Phase 4, test 4) plus the
budget typo-catcher (test 6's unit form).

The join is structural — index present / in range / unique, enum valid —
so unlike the Phase 3 regex-over-prose guard there is no false-positive
surface to tune. What it cannot check is summary faithfulness; that is a
documented open gap, not a missed assertion here.
"""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from app.agent.researcher import UsageSummary
from app.agent.trading.domain.errors import VendorError
from app.agent.trading.infrastructure import news_digest_port
from app.agent.trading.infrastructure.news_digest_port import (
    BATCH_SIZE,
    _assert_within_budget,
    _join,
    _parse_index,
    _render_batch,
    build_digest,
)


def _articles(n: int) -> list[dict]:
    return [
        {
            "headline": f"headline {i}",
            "_pub_date": date(2025, 3, 10),
            "source": "wire",
            "url": f"https://example.com/{i}",
            "summary": f"body {i}",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Test 4 — batch integrity
# ---------------------------------------------------------------------------

def test_join_flags_missing_duplicate_and_invalid_enum():
    articles = _articles(5)
    parsed = [
        {"index": 0, "summary": "s0", "sentiment": "positive", "relevance": "primary"},
        {"index": 1, "summary": "s1", "sentiment": "negative", "relevance": "primary"},
        {"index": 1, "summary": "s1 again", "sentiment": "positive", "relevance": "primary"},  # duplicate
        {"index": 2, "summary": "s2", "sentiment": "bullish", "relevance": "primary"},  # invalid enum
        {"index": 4, "summary": "s4", "sentiment": "neutral", "relevance": "mentioned"},
        # index 3 missing entirely — data loss, must be visible
    ]

    items, issues = _join(articles, parsed)

    assert any("duplicate index 1" in i for i in issues)
    assert any("missing index 3" in i for i in issues)
    assert any("invalid sentiment 'bullish'" in i for i in issues)

    by_headline = {i.headline: i for i in items}
    assert set(by_headline) == {"headline 0", "headline 1", "headline 2", "headline 4"}
    # invalid enum degrades to neutral rather than being dropped or trusted
    assert by_headline["headline 2"].sentiment == "neutral"
    # the duplicate's first occurrence wins
    assert by_headline["headline 1"].summary == "s1"
    # no NewsItem carries metadata that isn't in the input set — the model
    # cannot introduce a headline, date, source, or URL
    input_headlines = {a["headline"] for a in articles}
    assert all(i.headline in input_headlines for i in items)
    assert all(i.published_date == date(2025, 3, 10) for i in items)


def test_join_validates_relevance_and_degrades_to_mentioned():
    """Relevance gets the same structural treatment as sentiment. It degrades
    to the middle value, not to either extreme: "unrelated" would drop a real
    article out of the sentiment aggregate over a formatting slip, and
    "primary" would admit noise into it."""
    articles = _articles(3)
    parsed = [
        {"index": 0, "summary": "s0", "sentiment": "positive", "relevance": "PRIMARY"},
        {"index": 1, "summary": "s1", "sentiment": "neutral", "relevance": "sort of"},
        {"index": 2, "summary": "s2", "sentiment": "neutral"},   # field absent
    ]

    items, issues = _join(articles, parsed)

    assert len(items) == 3
    assert items[0].relevance == "primary"          # case-normalized, no issue
    assert items[1].relevance == "mentioned"
    assert items[2].relevance == "mentioned"
    assert any("invalid relevance 'sort of'" in i for i in issues)
    assert any("invalid relevance ''" in i for i in issues)
    assert not any("index 0" in i for i in issues)


def test_render_batch_names_the_company_under_analysis():
    """Without this the model scores each article against whichever company
    the article is about — which is how a Netflix sell-off became a data
    point in MSFT's aggregate on the first live run."""
    rendered = _render_batch(_articles(2), "msft")
    assert "COMPANY UNDER ANALYSIS: MSFT" in rendered
    assert rendered.index("COMPANY UNDER ANALYSIS") < rendered.index("[0] HEADLINE")


def test_bracketed_index_is_accepted_not_dropped():
    """Regression, from the first live run (FIG, 2026-08-21): Haiku returned
    "index": "[0]" for a 3-article batch — echoing the prompt's own [N]
    marker — and all three articles were dropped as unparseable. They were
    the most on-topic stories in the batch, so the cost of rejecting an
    unambiguous form is real article loss."""
    articles = _articles(3)
    parsed = [
        {"index": "[0]", "summary": "s0", "sentiment": "neutral", "relevance": "primary"},
        {"index": "[1]", "summary": "s1", "sentiment": "positive", "relevance": "primary"},
        {"index": " [2] ", "summary": "s2", "sentiment": "negative", "relevance": "primary"},
    ]

    items, issues = _join(articles, parsed)

    assert issues == []
    assert [i.headline for i in items] == ["headline 0", "headline 1", "headline 2"]
    assert [i.sentiment for i in items] == ["neutral", "positive", "negative"]


def test_a_batch_wrapped_in_an_extra_array_is_unwrapped_not_dropped():
    """Regression, from the ASML run on deepseek-v4-flash (2026-08-29): one
    batch came back as [[{...}, ...]] instead of [{...}, ...].

    The outer list passes the "is it an array" check in `_summarize_batch`,
    so it reached the join, where `obj["index"]` on a list raises TypeError.
    Every article in the batch was flagged unparseable and dropped — seven
    real stories — and the digest then scored ASML off the five that
    survived elsewhere. Same reasoning as the bracketed-index case above:
    the nesting is unambiguous, so recovering it beats losing articles the
    model summarised correctly."""
    articles = _articles(3)
    inner = [
        {"index": 0, "summary": "s0", "sentiment": "neutral", "relevance": "primary"},
        {"index": 1, "summary": "s1", "sentiment": "positive", "relevance": "primary"},
        {"index": 2, "summary": "s2", "sentiment": "negative", "relevance": "primary"},
    ]

    items, issues = _join(articles, [inner])

    assert issues == []
    assert [i.headline for i in items] == ["headline 0", "headline 1", "headline 2"]


def test_only_one_level_of_nesting_is_unwrapped():
    """A deeper structure is a genuinely malformed response, not the known
    one-level wrap, and must still be flagged rather than ride in on the
    recovery."""
    articles = _articles(2)
    obj = {"index": 0, "summary": "s0", "sentiment": "neutral", "relevance": "primary"}

    items, issues = _join(articles, [[[obj]]])

    assert items == []
    assert any("unparseable index" in i for i in issues)
    assert any("missing index 0" in i for i in issues)


def test_partial_nesting_recovers_every_article():
    """The wrap has been seen covering only part of a batch."""
    articles = _articles(3)
    flat = {"index": 0, "summary": "s0", "sentiment": "neutral", "relevance": "primary"}
    wrapped = [
        {"index": 1, "summary": "s1", "sentiment": "neutral", "relevance": "primary"},
        {"index": 2, "summary": "s2", "sentiment": "neutral", "relevance": "primary"},
    ]

    items, issues = _join(articles, [flat, wrapped])

    assert issues == []
    assert len(items) == 3


def test_parse_index_accepts_unambiguous_forms_and_rejects_guesswork():
    assert _parse_index(3) == 3
    assert _parse_index("3") == 3
    assert _parse_index("[3]") == 3
    assert _parse_index(" [3] ") == 3
    assert _parse_index(3.0) == 3

    # a bool must not become index 1 via int(True); a non-integral float must
    # not silently truncate onto the wrong article
    with pytest.raises(ValueError):
        _parse_index(True)
    with pytest.raises(ValueError):
        _parse_index(1.7)
    with pytest.raises(ValueError):
        _parse_index("first")
    with pytest.raises(TypeError):
        _parse_index(None)


def test_join_flags_out_of_range_and_unparseable_indices():
    articles = _articles(2)
    parsed = [
        {"index": 0, "summary": "ok", "sentiment": "neutral", "relevance": "primary"},
        {"index": 99, "summary": "phantom", "sentiment": "positive", "relevance": "primary"},
        {"summary": "no index at all", "sentiment": "positive", "relevance": "primary"},
        {"index": "one", "summary": "bad index", "sentiment": "positive", "relevance": "primary"},
    ]

    items, issues = _join(articles, parsed)

    assert len(items) == 1
    assert any("out of range" in i for i in issues)
    assert sum("unparseable index" in i for i in issues) == 2
    assert any("missing index 1" in i for i in issues)


def test_render_batch_numbers_articles_and_truncates_bodies():
    articles = _articles(2)
    articles[1]["summary"] = "x" * 2000
    rendered = _render_batch(articles, "ACN")
    assert "[0] HEADLINE: headline 0" in rendered
    assert "[1] HEADLINE: headline 1" in rendered
    assert "x" * 600 in rendered
    assert "x" * 601 not in rendered


# ---------------------------------------------------------------------------
# build_digest orchestration (LLM monkeypatched)
# ---------------------------------------------------------------------------

def _fake_usage(in_tok: int = 100, out_tok: int = 50):
    class _U:
        input_tokens = in_tok
        output_tokens = out_tok
        cache_creation_input_tokens = 0
        cache_read_input_tokens = 0
    return _U()


@pytest.mark.anyio
async def test_build_digest_batches_and_logs_cost_once(monkeypatch):
    articles = _articles(BATCH_SIZE + 3)   # forces exactly two batches
    batch_sizes: list[int] = []
    log_calls: list[tuple] = []

    async def fake_summarize(client, batch, ticker):
        batch_sizes.append(len(batch))
        return (
            [{"index": i, "summary": f"s{i}", "sentiment": "neutral", "relevance": "primary"}
             for i in range(len(batch))],
            _fake_usage(),
        )

    def fake_log_cost(ticker, mode, usage, *args, **kwargs):
        log_calls.append((ticker, mode, usage.input_tokens, usage.output_tokens))
        return 0.0123

    monkeypatch.setattr(news_digest_port, "_summarize_batch", fake_summarize)
    monkeypatch.setattr(news_digest_port, "log_cost", fake_log_cost)

    items, issues, cost, _cost_event, _flags = await build_digest(articles, "ACN")

    # a multiset, not a sequence: batches run concurrently, so the order they
    # start in is not a property worth pinning
    assert sorted(batch_sizes) == sorted([BATCH_SIZE, 3])
    assert len(items) == BATCH_SIZE + 3
    assert issues == []
    # one summed log line per run, not one per batch
    assert log_calls == [("ACN", "trading-news", 200, 100)]
    assert cost == 0.0123


@pytest.mark.anyio
async def test_item_order_follows_input_not_completion_order(monkeypatch):
    """filter_and_dedup hands over articles newest-first and the digest must
    keep that order, so truncation and the report both stay meaningful. With
    concurrent batches, completion order no longer matches input order — here
    the later batch finishes first, deliberately."""
    articles = _articles(BATCH_SIZE + 3)

    async def fake_summarize(client, batch, ticker):
        # the small trailing batch returns immediately; the first batch yields
        # to the event loop first, so it completes last
        if len(batch) == BATCH_SIZE:
            await asyncio.sleep(0.02)
        return (
            [{"index": i, "summary": batch[i]["headline"], "sentiment": "neutral",
              "relevance": "primary"} for i in range(len(batch))],
            _fake_usage(),
        )

    monkeypatch.setattr(news_digest_port, "_summarize_batch", fake_summarize)
    monkeypatch.setattr(news_digest_port, "log_cost", lambda *a, **k: 0.01)

    items, issues, _, _cost_event, _flags = await build_digest(articles, "ACN")

    assert issues == []
    assert [i.headline for i in items] == [a["headline"] for a in articles]


@pytest.mark.anyio
async def test_one_unparseable_batch_costs_its_articles_not_the_run(monkeypatch):
    """A malformed reply to one batch must not discard the other batches'
    articles — or their spend. The loss is recorded as an issue rather than
    passing silently as a shorter digest."""
    articles = _articles(BATCH_SIZE + 3)

    async def fake_summarize(client, batch, ticker):
        if len(batch) == 3:
            raise VendorError("Haiku returned non-JSON digest")
        return (
            [{"index": i, "summary": f"s{i}", "sentiment": "neutral", "relevance": "primary"}
             for i in range(len(batch))],
            _fake_usage(),
        )

    monkeypatch.setattr(news_digest_port, "_summarize_batch", fake_summarize)
    monkeypatch.setattr(news_digest_port, "log_cost", lambda *a, **k: 0.01)

    items, issues, cost, _cost_event, _flags = await build_digest(articles, "ACN")

    assert len(items) == BATCH_SIZE          # the surviving batch is intact
    assert cost == 0.01
    assert len(issues) == 1
    assert "batch of 3 article(s) failed" in issues[0]
    assert "absent from this digest" in issues[0]
    assert "headline 15" in issues[0]        # names where the hole is


@pytest.mark.anyio
async def test_every_batch_failing_raises_instead_of_looking_like_no_news(monkeypatch):
    """An empty digest is a legitimate result for a quiet ticker, so it must
    not also be what a total failure produces."""
    async def always_fail(client, batch, ticker):
        raise VendorError("Haiku returned non-JSON digest")

    monkeypatch.setattr(news_digest_port, "_summarize_batch", always_fail)

    with pytest.raises(VendorError, match="refusing to return an empty digest"):
        await build_digest(_articles(BATCH_SIZE + 3), "ACN")


@pytest.mark.anyio
async def test_concurrency_is_bounded(monkeypatch):
    """A full-cap run is 20 batches; they must not all be issued at once."""
    articles = _articles(BATCH_SIZE * 8)
    in_flight = 0
    peak = 0

    async def fake_summarize(client, batch, ticker):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return (
            [{"index": i, "summary": f"s{i}", "sentiment": "neutral", "relevance": "primary"}
             for i in range(len(batch))],
            _fake_usage(),
        )

    monkeypatch.setattr(news_digest_port, "_summarize_batch", fake_summarize)
    monkeypatch.setattr(news_digest_port, "log_cost", lambda *a, **k: 0.01)

    await build_digest(articles, "ACN")

    assert peak <= news_digest_port.MAX_CONCURRENT_BATCHES
    assert peak > 1, "batches ran serially — the concurrency is not doing anything"


@pytest.mark.anyio
async def test_build_digest_empty_input_skips_llm_entirely(monkeypatch):
    async def explode(client, batch, ticker):
        raise AssertionError("LLM called for an empty article list")

    monkeypatch.setattr(news_digest_port, "_summarize_batch", explode)

    items, issues, cost, cost_event, flags = await build_digest([], "ACN")

    assert items == [] and issues == [] and cost is None
    assert cost_event is None and flags == []


# ---------------------------------------------------------------------------
# Test 6 (unit form) — budget typo-catcher
# ---------------------------------------------------------------------------

def test_budget_guard_trips_over_threshold_and_passes_under_it():
    _assert_within_budget(None)      # pricing unconfigured: nothing to check
    _assert_within_budget(0.02)      # a normal run, ~10x under
    with pytest.raises(AssertionError, match="exceeds"):
        _assert_within_budget(0.50)  # a Haiku run cannot cost this — wrong model
