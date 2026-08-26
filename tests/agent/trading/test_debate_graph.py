"""The debate CYCLE, end to end, offline.

Two things this covers that the router tests cannot: that the router is
actually wired to the edges, and that the add-reducer accumulates one turn
per super-step rather than doubling. Both run against an in-memory
checkpointer with the LLM stubbed, so they need no database, no API key, and
no money — which is what makes them safe to run on every commit, unlike the
five live runs they stand in front of.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import app.agent.trading.application.debate_nodes as debate_nodes
import app.agent.trading.application.nodes as nodes
import app.agent.trading.application.risk_nodes as risk_nodes
from app.agent.trading.application.debate_router import MAX_ROUNDS, MAX_TURNS
from app.agent.trading.domain.debate import DebateClaim, DebateTurn, DebateTurnPayload
from app.agent.trading.domain.decision_memo import DecisionMemo, Verdict
from app.agent.trading.domain.fundamentals_report import FundamentalsReport
from app.agent.trading.domain.risk import PERSONAS, RiskTurn, RiskTurnPayload
from app.agent.trading.infrastructure.graph import build_trading_graph

AS_OF = date(2026, 8, 19)


def _stub_risk_panel(monkeypatch):
    """Stubbed at the port seam, same posture as _stub_debate: the risk
    router, both layer-3 assert blocks and the reducer all still run for
    real, only the network call is replaced."""
    async def fake_turn(state, persona, turn_index):
        return RiskTurn(
            turn_index=turn_index,
            round_num=(turn_index // len(PERSONAS)) + 1,
            persona=persona,
            payload=RiskTurnPayload(argument=f"{persona} argument {turn_index}"),
            estimated_cost_usd=0.01,
        )

    monkeypatch.setattr(risk_nodes, "run_risk_turn", fake_turn)


def _stub_synthesis(monkeypatch):
    """Stubbed at the same seam nodes.synthesizer_node calls through — this
    test file is about the debate/risk CYCLES, not the synthesis LLM call,
    so the memo it produces is built directly from the caveats
    synthesizer_node already computed in Python, with no network involved."""
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


def _stub_debate(monkeypatch, *, productive=True, cost=0.01):
    """Stubbed at the port seam, so the router, both layer-3 assert blocks,
    the reducer and the per-turn checkpoint all still run for real."""
    async def fake_turn(state, side, turn_index):
        return DebateTurn(
            turn_index=turn_index,
            round_num=(turn_index // 2) + 1,
            side=side,
            payload=DebateTurnPayload(
                stance="hold",
                argument=f"{side} argument {turn_index}",
                claims=[
                    DebateClaim(
                        claim_id=f"c{turn_index}" if productive else "c0",
                        text="stub",
                        evidence_ref="none",
                    )
                ],
            ),
            productive=productive,
            estimated_cost_usd=cost,
        )

    monkeypatch.setattr(debate_nodes, "run_debate_turn", fake_turn)


def _stub_fundamentals(monkeypatch):
    async def fake(ticker: str):
        return FundamentalsReport(
            ticker=ticker,
            summary="# Stub memo",
            input_tokens=0,
            cache_write_tokens=0,
            cache_read_tokens=0,
            output_tokens=0,
            generated_at=AS_OF,
        )

    monkeypatch.setattr(nodes, "get_fundamentals_report", fake)


async def _run(graph, ticker="ACN"):
    config = {"configurable": {"thread_id": f"debate-graph-{uuid.uuid4()}"}}
    result = await graph.ainvoke({"ticker": ticker, "as_of_date": AS_OF}, config=config)
    return result, config


async def _history(graph, config) -> list:
    return [state async for state in graph.aget_state_history(config)]


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_the_debate_entry_edge_is_conditional():
    """A plain edge into bull_turn would execute one turn before any router
    ever saw the state, and the no-evidence skip could never fire."""
    graph = build_trading_graph(InMemorySaver()).get_graph()
    entry = [e for e in graph.edges if e.source == "sentiment"]

    assert {e.target for e in entry} == {"bull_turn", "bear_turn", "debate_close"}
    assert all(e.conditional for e in entry)


def test_both_debate_nodes_carry_the_full_route_map():
    """Including the bull -> bull branch that correct alternation never takes.
    If it ever fires you want a visible loop in the checkpoint history that
    the alternation assert then names precisely — not a KeyError deep in
    LangGraph routing that reads as a framework bug."""
    graph = build_trading_graph(InMemorySaver()).get_graph()

    for src in ("bull_turn", "bear_turn"):
        targets = {e.target for e in graph.edges if e.source == src}
        assert targets == {"bull_turn", "bear_turn", "debate_close"}


def test_the_risk_panel_entry_edge_is_conditional():
    """Same reasoning as the debate's own entry edge, one cycle up: a plain
    edge into neutral_turn would run one turn before any router saw the
    state, and risk_router's no-debate skip could never fire."""
    graph = build_trading_graph(InMemorySaver()).get_graph()
    entry = [e for e in graph.edges if e.source == "debate_close"]

    assert {e.target for e in entry} == {"neutral_turn", "aggressive_turn", "conservative_turn", "risk_close"}
    assert all(e.conditional for e in entry)


def test_all_three_risk_nodes_carry_the_full_route_map():
    graph = build_trading_graph(InMemorySaver()).get_graph()

    for src in ("neutral_turn", "aggressive_turn", "conservative_turn"):
        targets = {e.target for e in graph.edges if e.source == src}
        assert targets == {"neutral_turn", "aggressive_turn", "conservative_turn", "risk_close"}


def test_the_post_risk_chain_is_still_linear():
    graph = build_trading_graph(InMemorySaver()).get_graph()
    linear = {(e.source, e.target) for e in graph.edges if not e.conditional}

    assert ("risk_close", "synthesizer") in linear
    assert ("synthesizer", "__end__") in linear
    # and debate_close -> risk is NOT linear — it's the risk panel's
    # conditional entry edge, asserted above
    assert ("debate_close", "risk") not in linear


def test_the_old_debate_and_risk_stub_nodes_are_gone():
    """Pre-Phase-5 threads are unresumable because "debate" no longer exists
    in the graph; pre-Phase-6 threads (including any checkpointed against
    Phase 5's single-node risk stub) are unresumable because "risk" no
    longer does either. Fresh --thread-id for all Phase 5/6 work."""
    graph = build_trading_graph(InMemorySaver()).get_graph()
    assert "debate" not in {n for n in graph.nodes}
    assert "risk" not in {n for n in graph.nodes}


# ---------------------------------------------------------------------------
# Termination, through the real edges
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_a_productive_debate_stops_at_the_round_cap(monkeypatch):
    _stub_debate(monkeypatch, productive=True)
    _stub_fundamentals(monkeypatch)
    _stub_risk_panel(monkeypatch)
    _stub_synthesis(monkeypatch)

    result, _ = await _run(build_trading_graph(InMemorySaver(), analysts=["fundamentals"]))

    turns = result["debate_turns"]
    assert len(turns) == MAX_TURNS
    assert [t.turn_index for t in turns] == list(range(MAX_TURNS))
    assert [t.side for t in turns] == ["bull", "bear"] * MAX_ROUNDS
    assert [t.round_num for t in turns] == [
        (i // 2) + 1 for i in range(MAX_TURNS)
    ]
    assert result["debate_terminated_by"] == "round_cap"


@pytest.mark.anyio
async def test_an_all_unproductive_debate_still_runs_to_the_cap(monkeypatch):
    """Regression for the removed early stop (2026-08-24). Across every
    debate measured, a turn's claims were reused at most ~25%, and the old
    clause needed BOTH of two consecutive turns at zero new claims — a
    conjunction that never occurred. Every stub turn here scores
    productive=False, which used to stop the debate after 2 turns; it must
    now run the full six and hit round_cap like any other debate, because
    MAX_ROUNDS is the only lever left."""
    _stub_debate(monkeypatch, productive=False)
    _stub_fundamentals(monkeypatch)
    _stub_risk_panel(monkeypatch)
    _stub_synthesis(monkeypatch)

    result, _ = await _run(build_trading_graph(InMemorySaver(), analysts=["fundamentals"]))

    assert len(result["debate_turns"]) == MAX_TURNS
    assert all(not t.productive for t in result["debate_turns"])
    assert result["debate_terminated_by"] == "round_cap"


@pytest.mark.anyio
async def test_the_reducer_accumulates_one_turn_per_super_step(monkeypatch):
    """Returning the accumulated list instead of a delta doubles it every
    step — a failure that looks exactly like a runaway loop."""
    _stub_debate(monkeypatch, productive=True)
    _stub_fundamentals(monkeypatch)
    _stub_risk_panel(monkeypatch)
    _stub_synthesis(monkeypatch)
    saver = InMemorySaver()
    graph = build_trading_graph(saver, analysts=["fundamentals"])

    _, config = await _run(graph)
    lengths = [
        len(state.values.get("debate_turns") or [])
        for state in await _history(graph, config)
    ]

    # every distinct length from 0 to the cap appears; no jump from 2 to 4
    assert set(lengths) == set(range(MAX_TURNS + 1))


@pytest.mark.anyio
async def test_the_run_checkpoints_after_every_turn(monkeypatch):
    """The whole reason a turn is a NODE and not a loop iteration: LangGraph
    persists at the end of every super-step, so one node per turn gives
    2 x rounds recovery points instead of one."""
    _stub_debate(monkeypatch, productive=True)
    _stub_fundamentals(monkeypatch)
    _stub_risk_panel(monkeypatch)
    _stub_synthesis(monkeypatch)
    saver = InMemorySaver()
    graph = build_trading_graph(saver, analysts=["fundamentals"])

    _, config = await _run(graph)

    # History is newest-first, so the first snapshot seen at each transcript
    # length is the one that closed that super-step.
    by_length: dict[int, object] = {}
    for state in await _history(graph, config):
        by_length.setdefault(len(state.values.get("debate_turns") or []), state)

    # every intermediate length has its own recovery point — a single
    # checkpoint at the end of a loop would give exactly two, 0 and the cap
    assert set(by_length) == set(range(MAX_TURNS + 1))

    # and each mid-debate one knows which turn was in flight, which is what
    # makes "resumes from the last per-round checkpoint" mean anything
    for length in range(MAX_TURNS):
        assert by_length[length].next == (
            ("bull_turn",) if length % 2 == 0 else ("bear_turn",)
        )


@pytest.mark.anyio
async def test_an_interrupt_mid_cycle_resumes_at_the_turn_in_flight(monkeypatch):
    """`interrupt_after` on a CYCLIC node stops after EVERY execution of it,
    so this is three interrupts, not one. Read three as three."""
    _stub_debate(monkeypatch, productive=True)
    _stub_fundamentals(monkeypatch)
    _stub_risk_panel(monkeypatch)
    _stub_synthesis(monkeypatch)
    saver = InMemorySaver()
    graph = build_trading_graph(
        saver, interrupt_after=["bull_turn"], analysts=["fundamentals"]
    )
    config = {"configurable": {"thread_id": f"debate-interrupt-{uuid.uuid4()}"}}

    await graph.ainvoke({"ticker": "ACN", "as_of_date": AS_OF}, config=config)
    snapshot = await graph.aget_state(config)

    # values first, then next: `next` alone cannot distinguish a never-run
    # thread from a completed one
    assert len(snapshot.values["debate_turns"]) == 1
    assert snapshot.next == ("bear_turn",)
    before = list(snapshot.values["debate_turns"])

    # resume on a graph without the interrupt — the turns already committed
    # must come back field-identical, not re-run
    resumed = build_trading_graph(saver, analysts=["fundamentals"])
    final = await resumed.ainvoke(None, config=config)

    assert len(final["debate_turns"]) == MAX_TURNS
    assert final["debate_turns"][0] == before[0]
    assert [t.turn_index for t in final["debate_turns"]] == list(range(MAX_TURNS))


# ---------------------------------------------------------------------------
# The skip path
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_a_run_with_no_analyst_evidence_skips_the_debate_entirely(monkeypatch):
    """Two models arguing from an empty pack reads like a debate and is
    theatre. The memo has to say so rather than carry the transcript."""
    _stub_debate(monkeypatch, productive=True)
    _stub_synthesis(monkeypatch)

    async def no_report(ticker: str):
        return None

    monkeypatch.setattr(nodes, "get_fundamentals_report", no_report)
    result, _ = await _run(build_trading_graph(InMemorySaver(), analysts=["fundamentals"]))

    assert result.get("debate_turns") in (None, [])
    assert result["debate_terminated_by"] == "no_evidence"
    assert any(
        "no debate took place (no_evidence)" in gap
        for gap in result["decision_memo"].data_gaps
    )


@pytest.mark.anyio
async def test_a_completed_debate_reaches_the_memo_as_evidence(monkeypatch):
    _stub_debate(monkeypatch, productive=True)
    _stub_fundamentals(monkeypatch)
    _stub_risk_panel(monkeypatch)
    _stub_synthesis(monkeypatch)

    result, _ = await _run(build_trading_graph(InMemorySaver(), analysts=["fundamentals"]))
    memo = result["decision_memo"]

    assert any(f"{MAX_TURNS}-turn bull/bear debate" in e for e in memo.evidence)
    assert any("round cap rather than resolving" in g for g in memo.data_gaps)
