"""Checkpoint fidelity for the custom domain types in TradingState.

`ALLOWED_MSGPACK_MODULES` is what restricts checkpoint deserialization to known
types. The failure it guards against is real but delayed: a new domain type
lands in TradingState, everything works in-process while the object is still in
memory, and the break only surfaces when a *different* process reads that
checkpoint back out of Postgres.

Note the enforcement comes from passing `allowed_msgpack_modules` explicitly,
NOT from the LANGGRAPH_STRICT_MSGPACK environment variable. That variable is
read once into a module-level constant when langgraph is first imported, so
checkpointer.py setting it afterwards has no effect, and langgraph consults it
only for serializers built *without* an explicit allowlist. Asserting the
variable would therefore prove nothing about whether anything is enforced —
`test_build_serde_blocks_a_type_outside_the_allowlist` checks the behaviour
instead.

Two tiers here, because they have different costs:

  * The serde tests need no database and run everywhere. They cover the actual
    regression — a domain type reachable from TradingState that nobody
    registered — and fail the moment a future phase adds one.
  * The graph test needs Postgres and skips without it. It exercises the whole
    path (real graph, real checkpointer, real Postgres) across two separate
    checkpointer connections.

Interrupting the graph is done with LangGraph's own `interrupt_after` rather
than an OS signal. The stub nodes downstream of `technical` are print
statements with nothing to await, so they all complete within microseconds of
one another — there is no wall-clock window in which a Ctrl+C could land
between them, which makes process-killing untestable rather than merely
awkward. `interrupt_after` stops at a node boundary deterministically.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import date
from enum import Enum
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import pandas as pd
import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

import app.agent.trading.application.debate_nodes as debate_nodes
import app.agent.trading.application.nodes as nodes
import app.agent.trading.application.risk_nodes as risk_nodes
from app.agent.trading.application.risk_router import RISK_MAX_TURNS
from app.agent.trading.domain.debate import DebateClaim, DebateTurn, DebateTurnPayload
from app.agent.trading.domain.decision_memo import DecisionMemo, Verdict
from app.agent.trading.domain.fundamentals_report import FundamentalsReport
from app.agent.trading.domain.news_digest import NewsDigest, NewsItem, SentimentSummary
from app.agent.trading.domain.risk import PERSONAS, RiskTurn, RiskTurnPayload
from app.agent.trading.domain.technical_report import TechnicalIndicators, TechnicalReport
from app.agent.trading.domain.trading_state import TradingState
from app.agent.trading.infrastructure.checkpointer import (
    ALLOWED_MSGPACK_MODULES,
    DB_URI,
    build_checkpointer,
    build_serde,
)
from app.agent.trading.infrastructure.graph import build_trading_graph

FIXTURE = Path(__file__).resolve().parents[3] / "tests/fixtures/avgo_ohlcv_sample.csv"


def _sample_report() -> TechnicalReport:
    """Includes a None indicator and a populated flagged-numbers list so the
    round-trip covers the optional and collection fields too, not just floats."""
    return TechnicalReport(
        ticker="ACN",
        as_of_date=date(2026, 8, 19),
        data_source="yfinance",
        bars_used=252,
        indicators=TechnicalIndicators(
            sma_50=330.1245,
            sma_200=None,
            rsi_14=41.2033,
            macd=-2.4471,
            macd_signal=-1.2313,
            macd_histogram=-1.2157936592253513,
            last_close=327.55,
            volume_vs_20d_avg=0.8842,
        ),
        interpretation="Momentum is bearish; the histogram sits at around -1.22.",
        interpretation_flagged_numbers=["812"],
    )


def _sample_digest() -> NewsDigest:
    """One populated item and every provenance counter non-default, so the
    round-trip covers the nested list-of-models path and the bool/int fields."""
    return NewsDigest(
        ticker="ACN",
        as_of_date=date(2026, 8, 19),
        window_start=date(2026, 8, 5),
        items=[
            NewsItem(
                headline="Acme beats estimates",
                published_date=date(2026, 8, 18),
                source="reuters",
                url="https://example.com/1",
                summary="Acme reported quarterly results above expectations.",
                sentiment="positive",
                relevance="primary",
            ),
            NewsItem(
                headline="Sector roundup",
                published_date=date(2026, 8, 12),
                source="yahoo",
                url="https://example.com/2",
                summary="Routine sector coverage.",
                sentiment="neutral",
                relevance="mentioned",
            ),
        ],
        raw_article_count=7,
        deduped_count=2,
        dropped_out_of_window=3,
        dropped_missing_date=1,
        truncated_by_cap=True,
    )


# ---------------------------------------------------------------------------
# Tier 1 — serde only. No database.
# ---------------------------------------------------------------------------

def test_technical_report_survives_msgpack_roundtrip():
    """The registered-type path: dump and reload a TechnicalReport through
    build_serde() — the same serializer the checkpointer uses in production,
    not an equivalent rebuilt here."""
    serde = build_serde()
    report = _sample_report()

    restored = serde.loads_typed(serde.dumps_typed(report))

    assert restored == report
    assert isinstance(restored, TechnicalReport)
    assert isinstance(restored.indicators, TechnicalIndicators)
    # the fields most likely to degrade quietly: full float precision, a None
    # optional, a date, and a list
    assert restored.indicators.macd_histogram == -1.2157936592253513
    assert restored.indicators.sma_200 is None
    assert restored.as_of_date == date(2026, 8, 19)
    assert restored.interpretation_flagged_numbers == ["812"]


def test_full_trading_state_survives_msgpack_roundtrip():
    """A whole TradingState dict, as the checkpointer actually stores it —
    two custom report types nested in one payload."""
    serde = build_serde()
    state: TradingState = {
        "ticker": "ACN",
        "fundamentals_report": FundamentalsReport(
            ticker="ACN",
            summary="# Memo",
            input_tokens=1,
            cache_write_tokens=2,
            cache_read_tokens=3,
            output_tokens=4,
            generated_at=date(2026, 8, 19),
        ),
        "technical_report": _sample_report(),
    }

    restored = serde.loads_typed(serde.dumps_typed(state))

    assert restored == state
    assert restored["technical_report"].indicators.rsi_14 == 41.2033


def test_news_digest_survives_msgpack_roundtrip():
    """Phase 4's types through the production serializer: the nested
    list[NewsItem], the Literal sentiment, dates, and the provenance bools."""
    serde = build_serde()
    digest = _sample_digest()
    summary = SentimentSummary(
        ticker="ACN",
        as_of_date=date(2026, 8, 19),
        positive=1,
        negative=0,
        neutral=1,
        net_score=0.5,
        article_count=2,
        excluded_by_relevance=3,
    )

    restored_digest = serde.loads_typed(serde.dumps_typed(digest))
    restored_summary = serde.loads_typed(serde.dumps_typed(summary))

    assert restored_digest == digest
    assert isinstance(restored_digest, NewsDigest)
    assert all(isinstance(i, NewsItem) for i in restored_digest.items)
    assert restored_digest.items[0].sentiment == "positive"
    assert [i.relevance for i in restored_digest.items] == ["primary", "mentioned"]
    assert restored_summary.excluded_by_relevance == 3
    assert restored_digest.truncated_by_cap is True
    assert restored_summary == summary
    assert isinstance(restored_summary, SentimentSummary)


def _sample_debate_turn() -> DebateTurn:
    """Two levels of nesting, both populated: DebateTurn -> DebateTurnPayload
    -> DebateClaim. Two levels means two places the allowlist can be
    incomplete, and an unregistered type fails on DESERIALIZATION ONLY — so
    without this the first symptom would be a red resume in a live crash
    test, an hour after the green run that wrote the checkpoint."""
    return DebateTurn(
        turn_index=3,
        round_num=2,
        side="bear",
        payload=DebateTurnPayload(
            stance="concede",
            concession_trigger="margin-hold",
            argument="The amortization roll-off is real.",
            claims=[
                DebateClaim(
                    claim_id="vmware-amort-rolloff",
                    text="Amortization steps down next year.",
                    evidence_ref="fundamentals",
                    evidence_quote="amortization of acquired intangibles falls",
                ),
                DebateClaim(
                    claim_id="derived-pressure",
                    text="Taken together those imply margin pressure.",
                    evidence_ref="none",
                ),
            ],
            rebuts=["margin-hold", "growth-durable"],
        ),
        productive=False,
        guard_flags=["71.4"],
        unquoted_evidence=["derived-pressure"],
        input_tokens=6100,
        output_tokens=690,
        estimated_cost_usd=0.0287,
    )


def test_debate_turn_survives_msgpack_roundtrip():
    """Phase 5's types through the production serializer."""
    serde = build_serde()
    turn = _sample_debate_turn()

    restored = serde.loads_typed(serde.dumps_typed(turn))

    assert restored == turn
    assert isinstance(restored, DebateTurn)
    assert isinstance(restored.payload, DebateTurnPayload)
    assert all(isinstance(c, DebateClaim) for c in restored.payload.claims)
    assert restored.payload.claims[0].evidence_ref == "fundamentals"
    assert restored.payload.stance == "concede"
    assert restored.payload.rebuts == ["margin-hold", "growth-durable"]
    assert restored.guard_flags == ["71.4"]
    assert restored.productive is False
    assert restored.estimated_cost_usd == 0.0287


def test_the_accumulated_debate_transcript_survives_msgpack_roundtrip():
    """The channel as the checkpointer actually stores it: a list of turns
    under the add-reducer, not one turn on its own."""
    serde = build_serde()
    turns = [_sample_debate_turn() for _ in range(3)]
    for i, turn in enumerate(turns):
        turn.turn_index = i

    restored = serde.loads_typed(serde.dumps_typed({"debate_turns": turns}))

    assert [t.turn_index for t in restored["debate_turns"]] == [0, 1, 2]
    assert all(isinstance(t, DebateTurn) for t in restored["debate_turns"])


class _UnregisteredModel(BaseModel):
    value: int


def test_build_serde_blocks_a_type_outside_the_allowlist():
    """Proves the *production* serializer enforces the allowlist, not merely
    that registered types survive a round-trip.

    Without this, every test above would keep passing if build_serde() ever
    stopped passing `allowed_msgpack_modules`: langgraph's default in that
    case is permissive — it warns and allows unregistered types — so the
    registered types would still round-trip cleanly while the guard had
    silently become a no-op. This is the test that fails in that scenario.
    """
    serde = build_serde()

    restored = serde.loads_typed(serde.dumps_typed(_UnregisteredModel(value=7)))

    assert isinstance(restored, dict)
    assert restored == {"value": 7}


def test_unregistered_top_level_type_degrades_to_a_dict_without_raising():
    """Negative control, and a documented surprise: an unregistered type does
    not raise on load. It is logged as blocked and comes back as a plain
    dict, so a resumed run hands a dict to a node that expects a
    TechnicalReport and fails later with an AttributeError somewhere in the
    graph rather than a clear serialization error at the read.

    This is what makes the registration list easy to get wrong and its
    breakage hard to trace — and it is why the test above asserts
    `isinstance`, not just equality.
    """
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            entry for entry in ALLOWED_MSGPACK_MODULES if entry[1] != "TechnicalReport"
        ]
    )

    restored = serde.loads_typed(serde.dumps_typed(_sample_report()))

    assert isinstance(restored, dict)
    assert not isinstance(restored, TechnicalReport)


def test_unregistered_nested_type_still_round_trips():
    """The other half of the picture, and the reason the structural test
    below cannot be replaced by a round-trip: with TechnicalIndicators
    unregistered but its parent registered, the payload survives intact —
    pydantic revalidates the parent and rebuilds the child from its dict, so
    nothing observable breaks.

    Registering nested types is therefore belt-and-braces rather than load-
    bearing on this path. Asserted so that a future reader doesn't conclude
    from a passing round-trip that the registration list is complete.
    """
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            entry
            for entry in ALLOWED_MSGPACK_MODULES
            if entry[1] != "TechnicalIndicators"
        ]
    )
    report = _sample_report()

    restored = serde.loads_typed(serde.dumps_typed(report))

    assert restored == report
    assert isinstance(restored.indicators, TechnicalIndicators)


def _custom_types(annotation, acc: set) -> None:
    """Collect every pydantic model and enum reachable from an annotation,
    descending through generics (list[X], X | None) and nested model fields."""
    if get_origin(annotation) is not None:
        for arg in get_args(annotation):
            _custom_types(arg, acc)
        return
    if not isinstance(annotation, type):
        return
    if issubclass(annotation, BaseModel):
        if annotation in acc:
            return
        acc.add(annotation)
        for field in annotation.model_fields.values():
            _custom_types(field.annotation, acc)
    elif issubclass(annotation, Enum):
        acc.add(annotation)


def test_every_domain_type_reachable_from_trading_state_is_registered():
    """The structural guard, and the one that earns its keep over time.

    A round-trip test cannot cover this on its own: per the two tests above,
    an unregistered *top-level* type degrades to a dict (caught), while an
    unregistered *nested* type round-trips cleanly (invisible). Walking
    TradingState catches both, so a report type added in a later phase
    without a matching ALLOWED_MSGPACK_MODULES entry fails here immediately
    rather than in whichever process first resumes a checkpoint.

    Deliberately stricter than the serde path strictly requires, since it
    also demands nested types be registered. That costs one line per type
    and removes the need to reason about which position a type will appear
    in before trusting a checkpoint.
    """
    found: set = set()
    for annotation in get_type_hints(TradingState).values():
        _custom_types(annotation, found)

    registered = {tuple(entry) for entry in ALLOWED_MSGPACK_MODULES}
    missing = {(t.__module__, t.__name__) for t in found} - registered

    assert missing == set(), (
        f"these types are reachable from TradingState but absent from "
        f"ALLOWED_MSGPACK_MODULES: {sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# Tier 2 — the full graph against real Postgres. Skipped without a database.
# ---------------------------------------------------------------------------

def _postgres_reachable() -> bool:
    """Probe directly with a short timeout. The pool in build_checkpointer
    retries for 30s before giving up, which would stall the suite on every
    run in an environment that simply has no database."""
    if not DB_URI:
        return False
    try:
        import psycopg

        with psycopg.connect(DB_URI, connect_timeout=2):
            return True
    except Exception:
        return False


requires_postgres = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="needs the checkpoint Postgres at TRADING_CHECKPOINT_DB_URI",
)


@requires_postgres
def test_build_checkpointer_wires_in_the_allowlisted_serde():
    """build_serde() is proven to enforce the allowlist above; this confirms
    build_checkpointer actually hands that output to AsyncPostgresSaver
    rather than some other serde construction that merely looks equivalent.

    The two can drift independently: a refactor that dropped the argument
    (`serde=JsonPlusSerializer()`) would leave the whole suite green, because
    a permissive serializer round-trips *registered* types perfectly well —
    and registered types are all the round-trip tests exercise. The guard
    would be a silent no-op with nothing to notice it.

    Asserted through behaviour rather than by reading
    `serde._allowed_msgpack_modules`: that attribute is private, and
    langgraph normalizes the list into a set at construction, so comparing
    it to ALLOWED_MSGPACK_MODULES is both fragile across versions and false.
    """

    async def check():
        async with build_checkpointer() as checkpointer:
            serde = checkpointer.serde

            blocked = serde.loads_typed(serde.dumps_typed(_UnregisteredModel(value=7)))
            assert isinstance(blocked, dict), (
                "the checkpointer's serde admitted an unregistered type — "
                "allowed_msgpack_modules is not wired through build_serde()"
            )

            # and the same object still round-trips the registered types, so
            # this is enforcement rather than a serializer that breaks
            # everything equally
            report = _sample_report()
            assert serde.loads_typed(serde.dumps_typed(report)) == report

    asyncio.run(check())


def _stub_expensive_nodes(monkeypatch, tmp_path) -> None:
    """Replace the network and vault I/O, keep everything else real.

    compute_indicators still runs for real over the frozen 252-bar fixture, so
    the TechnicalReport being checkpointed carries genuine float values rather
    than round numbers that could mask a precision loss in the round-trip.
    """
    # utc=True is required, not incidental: the fixture spans a DST change, so
    # its timestamps carry mixed -04:00/-05:00 offsets and both `parse_dates`
    # and `format="ISO8601"` leave the index as strings. technical_node calls
    # df.index[-1].date(), which needs real Timestamps. Midnight-local to UTC
    # is a same-day shift at both offsets, so as_of_date is unaffected.
    df = pd.read_csv(FIXTURE, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)

    async def fake_fundamentals(ticker: str, run_id: str | None = None):
        return FundamentalsReport(
            ticker=ticker,
            summary="# Stub memo",
            input_tokens=0,
            cache_write_tokens=0,
            cache_read_tokens=0,
            output_tokens=0,
            generated_at=date(2026, 8, 19),
        )

    async def fake_price_history(ticker: str, as_of: date):
        return df, "fixture", 0

    async def fake_interpret(ticker: str, indicators, run_id: str | None = None):
        return "Stub interpretation, no numbers.", [], [], None, None

    # The news ports are stubbed at the same seam the real node calls, so
    # news_node's own body — the as_of guard, filter_and_dedup, the digest
    # assembly, the lookahead post-assert — still runs for real, and the
    # NewsDigest being checkpointed carries genuine filter-derived values.
    from datetime import datetime, timedelta, timezone

    from app.agent.trading.domain.news_digest import NewsItem

    async def fake_fetch_news(ticker: str, as_of: date, lookback_days: int = 14):
        raw = [
            {
                "headline": "Stub story one",
                "datetime": int(
                    datetime(
                        as_of.year, as_of.month, as_of.day, 12, tzinfo=timezone.utc
                    ).timestamp()
                ),
                "source": "stubwire",
                "url": "https://example.com/1",
                "summary": "body one",
            },
            {
                "headline": "Stub story two",
                "datetime": int(
                    datetime(
                        as_of.year, as_of.month, as_of.day, 12, tzinfo=timezone.utc
                    ).timestamp()
                )
                - 3 * 86400,
                "source": "stubwire",
                "url": "https://example.com/2",
                "summary": "body two",
            },
        ]
        return raw, as_of - timedelta(days=lookback_days)

    async def fake_build_digest(articles, ticker, run_id: str | None = None):
        items = [
            NewsItem(
                headline=a["headline"],
                published_date=a["_pub_date"],
                source=a["source"],
                url=a["url"],
                summary=f"summary of {a['headline']}",
                sentiment="positive" if i == 0 else "neutral",
                relevance="primary",
            )
            for i, a in enumerate(articles)
        ]
        return items, [], None, None, []

    monkeypatch.setattr(nodes, "get_fundamentals_report", fake_fundamentals)
    monkeypatch.setattr(nodes, "get_price_history", fake_price_history)
    monkeypatch.setattr(nodes, "interpret_indicators", fake_interpret)
    monkeypatch.setattr(
        nodes, "save_technical_report", lambda report, cost_usd=None: tmp_path / "stub.md"
    )
    monkeypatch.setattr(nodes, "fetch_company_news", fake_fetch_news)
    monkeypatch.setattr(nodes, "build_digest", fake_build_digest)

    # The debate is stubbed at the port seam, so the whole cycle — the
    # router, the alternation and cap asserts, the add-reducer, and the
    # checkpoint written after every turn — still runs for real. Only the
    # network call is replaced.
    async def fake_debate_turn(state, side, turn_index):
        return DebateTurn(
            turn_index=turn_index,
            round_num=(turn_index // 2) + 1,
            side=side,
            payload=DebateTurnPayload(
                stance="hold",
                argument=f"stub argument from {side}",
                claims=[
                    DebateClaim(
                        claim_id=f"{side}-{turn_index}",
                        text="stub claim",
                        evidence_ref="none",
                    )
                ],
                rebuts=[] if turn_index == 0 else [f"stub-{turn_index - 1}"],
            ),
            estimated_cost_usd=0.01,
        )

    monkeypatch.setattr(debate_nodes, "run_debate_turn", fake_debate_turn)

    # Same reason as the debate stub above, one cycle further: the risk
    # panel and the synthesizer both run for real after the debate now, and
    # both would otherwise make real network calls during what these tests
    # intend as an offline checkpoint round-trip.
    async def fake_risk_turn(state, persona, turn_index):
        return RiskTurn(
            turn_index=turn_index,
            round_num=(turn_index // len(PERSONAS)) + 1,
            persona=persona,
            payload=RiskTurnPayload(argument=f"stub argument from {persona}"),
            estimated_cost_usd=0.01,
        )

    monkeypatch.setattr(risk_nodes, "run_risk_turn", fake_risk_turn)

    async def fake_run_synthesis(state, *, ledger, base_gaps, base_evidence, as_of, client=None):
        technical = state.get("technical_report")
        return DecisionMemo(
            ticker=state["ticker"],
            bull_case="stub bull",
            bear_case="stub bear",
            research_thesis="stub thesis",
            risk_debate_summary="stub risk narrative",
            technical_signal=(
                technical.interpretation
                if technical is not None
                else "NOT RUN — technical analyst was excluded from this run"
            ),
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


@requires_postgres
def test_technical_report_survives_interrupt_and_a_fresh_checkpointer(
    monkeypatch, tmp_path
):
    """Phase 1 stops the graph after `technical` and closes the checkpointer.
    Phase 2 opens a *new* checkpointer connection and a *new* compiled graph,
    which is what forces an actual deserialize from Postgres rather than a
    read of an object still resident in memory — the process-A-writes,
    process-B-reads case ALLOWED_MSGPACK_MODULES protects.
    """
    _stub_expensive_nodes(monkeypatch, tmp_path)
    thread_id = f"test-checkpoint-roundtrip-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    async def phase_1():
        async with build_checkpointer() as checkpointer:
            graph = build_trading_graph(checkpointer, interrupt_after=["technical"])
            await graph.ainvoke(
                {"ticker": "ACN", "as_of_date": date(2026, 8, 19)}, config=config
            )
            snapshot = await graph.aget_state(config)
            return snapshot.values["technical_report"], snapshot.next

    async def phase_2():
        async with build_checkpointer() as checkpointer:
            graph = build_trading_graph(checkpointer)
            snapshot = await graph.aget_state(config)
            resumed = snapshot.values["technical_report"]
            final = await graph.ainvoke(None, config=config)
            return resumed, final

    async def cleanup():
        """Every run writes a thread to the real checkpoint database, so drop
        it afterwards — otherwise the dev DB accumulates one dead thread per
        test run, and a checkpoint table full of test rows is exactly the
        kind of noise that makes a real resume harder to inspect later."""
        async with build_checkpointer() as checkpointer:
            await checkpointer.adelete_thread(thread_id)

    try:
        before, next_nodes = asyncio.run(phase_1())
        # confirms the interrupt landed exactly where intended, not a node
        # early or late — without this the test could pass on a graph that
        # never ran
        assert next_nodes == ("news",)
        assert before is not None

        after, final = asyncio.run(phase_2())

        assert isinstance(after, TechnicalReport)
        assert after == before
        assert after.indicators.rsi_14 == before.indicators.rsi_14
        assert after.indicators.macd_histogram == before.indicators.macd_histogram
        assert after.bars_used == before.bars_used
        # the resumed run reaches the end and still sees the checkpointed report
        assert final["decision_memo"].technical_signal == before.interpretation
    finally:
        asyncio.run(cleanup())


@requires_postgres
def test_news_digest_survives_interrupt_and_a_fresh_checkpointer(
    monkeypatch, tmp_path
):
    """Phase 4's exit-criterion round-trip: stop after `news`, close the
    checkpointer, then read the NewsDigest back through a *new* connection
    and compiled graph — the process-A-writes, process-B-reads case that a
    missing ALLOWED_MSGPACK_MODULES entry breaks. The resume then runs the
    real sentiment_node over the deserialized digest, which would fail on
    the degraded-to-dict form an unregistered type comes back as."""
    _stub_expensive_nodes(monkeypatch, tmp_path)
    thread_id = f"test-news-checkpoint-roundtrip-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    async def phase_1():
        async with build_checkpointer() as checkpointer:
            graph = build_trading_graph(checkpointer, interrupt_after=["news"])
            await graph.ainvoke(
                {"ticker": "ACN", "as_of_date": date(2026, 8, 19)}, config=config
            )
            snapshot = await graph.aget_state(config)
            return snapshot.values["news_digest"], snapshot.next

    async def phase_2():
        async with build_checkpointer() as checkpointer:
            graph = build_trading_graph(checkpointer)
            snapshot = await graph.aget_state(config)
            resumed = snapshot.values["news_digest"]
            final = await graph.ainvoke(None, config=config)
            return resumed, final

    async def cleanup():
        async with build_checkpointer() as checkpointer:
            await checkpointer.adelete_thread(thread_id)

    try:
        before, next_nodes = asyncio.run(phase_1())
        assert next_nodes == ("sentiment",)
        assert isinstance(before, NewsDigest)
        assert len(before.items) == 2

        after, final = asyncio.run(phase_2())

        assert isinstance(after, NewsDigest)
        assert after == before
        assert all(isinstance(i, NewsItem) for i in after.items)
        assert after.items[0].published_date == date(2026, 8, 19)
        assert after.as_of_date == date(2026, 8, 19)

        # sentiment_node ran on the resumed side over the deserialized digest
        summary = final["sentiment_summary"]
        assert isinstance(summary, SentimentSummary)
        assert summary.article_count == 2
        assert (summary.positive, summary.neutral) == (1, 1)
        assert summary.net_score == 0.5
    finally:
        asyncio.run(cleanup())


@requires_postgres
def test_debate_transcript_survives_interrupt_and_a_fresh_checkpointer(
    monkeypatch, tmp_path
):
    """Phase 5's exit-criterion round-trip, and the reason a debate turn is a
    NODE rather than a loop iteration.

    Phase 1 stops after `bull_turn` — which on a cyclic node means after
    turn 0 — and closes the checkpointer. Phase 2 opens a NEW connection and
    a NEW compiled graph, forcing an actual deserialize of the nested
    DebateTurn -> DebateTurnPayload -> DebateClaim out of Postgres. Under the
    loop design there would be nothing to read here: a single checkpoint at
    node exit, and a kill mid-debate would resume at turn 0.
    """
    _stub_expensive_nodes(monkeypatch, tmp_path)
    thread_id = f"test-debate-checkpoint-roundtrip-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    async def phase_1():
        async with build_checkpointer() as checkpointer:
            graph = build_trading_graph(checkpointer, interrupt_after=["bull_turn"])
            await graph.ainvoke(
                {"ticker": "ACN", "as_of_date": date(2026, 8, 19)}, config=config
            )
            snapshot = await graph.aget_state(config)
            return snapshot.values.get("debate_turns"), snapshot.next

    async def phase_2():
        async with build_checkpointer() as checkpointer:
            graph = build_trading_graph(checkpointer)
            snapshot = await graph.aget_state(config)
            resumed = snapshot.values["debate_turns"]
            final = await graph.ainvoke(None, config=config)
            return resumed, final

    async def cleanup():
        async with build_checkpointer() as checkpointer:
            await checkpointer.adelete_thread(thread_id)

    try:
        # values before next: `next` alone cannot tell a never-run thread
        # from a completed one
        before, next_nodes = asyncio.run(phase_1())
        assert before is not None and len(before) == 1
        assert next_nodes == ("bear_turn",)

        after, final = asyncio.run(phase_2())

        # the checkpointed turn comes back as the real type, two levels deep,
        # rather than the plain dict an unregistered type degrades to
        assert isinstance(after[0], DebateTurn)
        assert isinstance(after[0].payload, DebateTurnPayload)
        assert all(isinstance(c, DebateClaim) for c in after[0].payload.claims)
        assert after[0] == before[0]

        # and the resumed run finishes the debate from where it stopped:
        # contiguous indices, strict alternation from bull, no re-run of
        # turn 0, and the termination reason recorded
        turns = final["debate_turns"]
        assert [t.turn_index for t in turns] == list(range(len(turns)))
        assert [t.side for t in turns] == [
            "bull" if i % 2 == 0 else "bear" for i in range(len(turns))
        ]
        assert turns[0] == before[0]
        assert final["debate_terminated_by"] in {"round_cap", "unproductive"}
    finally:
        asyncio.run(cleanup())


@requires_postgres
def test_risk_transcript_survives_interrupt_and_a_fresh_checkpointer(
    monkeypatch, tmp_path
):
    """Phase 6 exit criterion 7, and the test criterion 7's own description
    flags as the one most likely to fail late: TWO `operator.add` channels
    (`debate_turns` and `risk_turns`) accumulating in ONE thread is a
    configuration Phase 5's debate-only round-trip test never covered, and
    the failure mode — a pending write re-applied on resume — is a SILENT
    double-append that still looks like a plausible transcript, just one
    turn too long.

    Interrupts after `neutral_turn` — the debate has already fully
    accumulated its own turns by then (POST_DEBATE_NODES runs before the
    risk cycle even starts), so this is exactly the "crash mid-round-2 of
    the risk cycle, after debate_turns is already fully accumulated" case
    the exit criterion asks for. The two assertions that matter: risk_turns
    ends up exactly right, AND debate_turns is byte-identical to what it was
    before the interrupt — proving the two reducers didn't interfere.
    """
    _stub_expensive_nodes(monkeypatch, tmp_path)
    thread_id = f"test-risk-checkpoint-roundtrip-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    async def phase_1():
        async with build_checkpointer() as checkpointer:
            graph = build_trading_graph(checkpointer, interrupt_after=["neutral_turn"])
            await graph.ainvoke(
                {"ticker": "ACN", "as_of_date": date(2026, 8, 19)}, config=config
            )
            snapshot = await graph.aget_state(config)
            return (
                snapshot.values.get("debate_turns"),
                snapshot.values.get("risk_turns"),
                snapshot.next,
            )

    async def phase_2():
        async with build_checkpointer() as checkpointer:
            graph = build_trading_graph(checkpointer)
            snapshot = await graph.aget_state(config)
            resumed_debate = snapshot.values["debate_turns"]
            resumed_risk = snapshot.values["risk_turns"]
            final = await graph.ainvoke(None, config=config)
            return resumed_debate, resumed_risk, final

    async def cleanup():
        async with build_checkpointer() as checkpointer:
            await checkpointer.adelete_thread(thread_id)

    try:
        debate_before, risk_before, next_nodes = asyncio.run(phase_1())
        # the debate cycle already ran to completion before the risk cycle
        # was ever entered — this is the "after debate_turns is already
        # fully accumulated" precondition the criterion asks for
        assert debate_before is not None and len(debate_before) > 0
        assert risk_before is not None and len(risk_before) == 1
        assert next_nodes == ("aggressive_turn",)

        debate_resumed, risk_resumed, final = asyncio.run(phase_2())

        # the checkpointed risk turn comes back as the real type, two levels
        # deep, rather than the plain dict an unregistered type degrades to
        assert isinstance(risk_resumed[0], RiskTurn)
        assert isinstance(risk_resumed[0].payload, RiskTurnPayload)
        assert risk_resumed[0] == risk_before[0]
        # and debate_turns is UNCHANGED by the resume of a DIFFERENT
        # add-reducer channel — the actual point of this test
        assert debate_resumed == debate_before

        final_debate = final["debate_turns"]
        final_risk = final["risk_turns"]

        assert final_debate == debate_before   # never touched by the risk resume

        assert len(final_risk) == RISK_MAX_TURNS
        assert [t.turn_index for t in final_risk] == list(range(RISK_MAX_TURNS))
        assert [t.persona for t in final_risk] == [
            PERSONAS[i % len(PERSONAS)] for i in range(RISK_MAX_TURNS)
        ]
        assert final_risk[0] == risk_before[0]
        assert final["risk_terminated_by"] == "round_cap"
    finally:
        asyncio.run(cleanup())
