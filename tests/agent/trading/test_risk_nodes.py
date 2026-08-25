"""Node-level wiring for the risk panel: the layer-3 asserts, the delta
shape, and the close node — mirrors test_debate_nodes.py.
"""

from __future__ import annotations

import pytest

import app.agent.trading.application.risk_nodes as risk_nodes
from app.agent.trading.application.risk_router import RISK_MAX_TURNS
from app.agent.trading.domain.risk import PERSONAS, RiskTurn, RiskTurnPayload


def _turn(index: int, persona: str | None = None) -> RiskTurn:
    return RiskTurn(
        turn_index=index,
        round_num=(index // len(PERSONAS)) + 1,
        persona=persona or PERSONAS[index % len(PERSONAS)],
        payload=RiskTurnPayload(argument="stub"),
    )


def _patch_port(monkeypatch, calls: list | None = None):
    async def fake_run(state, persona, turn_index):
        if calls is not None:
            calls.append((persona, turn_index))
        return _turn(turn_index, persona)

    monkeypatch.setattr(risk_nodes, "run_risk_turn", fake_run)


@pytest.mark.anyio
async def test_neutral_turn_returns_a_single_element_delta(monkeypatch):
    _patch_port(monkeypatch)

    update = await risk_nodes.neutral_turn_node(
        {"ticker": "ACN", "risk_turns": [_turn(0), _turn(1), _turn(2)]}
    )

    assert list(update) == ["risk_turns"]
    assert len(update["risk_turns"]) == 1
    assert update["risk_turns"][0].turn_index == 3


@pytest.mark.anyio
async def test_turn_index_and_persona_come_from_the_transcript_length(monkeypatch):
    calls: list = []
    _patch_port(monkeypatch, calls)

    await risk_nodes.neutral_turn_node({"ticker": "ACN"})
    await risk_nodes.aggressive_turn_node({"ticker": "ACN", "risk_turns": [_turn(0)]})
    await risk_nodes.conservative_turn_node(
        {"ticker": "ACN", "risk_turns": [_turn(0), _turn(1)]}
    )

    assert calls == [("neutral", 0), ("aggressive", 1), ("conservative", 2)]


@pytest.mark.anyio
async def test_node_refuses_to_run_past_the_cap(monkeypatch):
    _patch_port(monkeypatch)
    state = {"ticker": "ACN", "risk_turns": [_turn(i) for i in range(RISK_MAX_TURNS)]}

    with pytest.raises(RuntimeError, match="router bypassed"):
        await risk_nodes.neutral_turn_node(state)


@pytest.mark.anyio
async def test_node_refuses_to_break_rotation(monkeypatch):
    _patch_port(monkeypatch)

    with pytest.raises(RuntimeError, match="Rotation broken"):
        await risk_nodes.aggressive_turn_node({"ticker": "ACN"})   # aggressive at index 0

    with pytest.raises(RuntimeError, match="Rotation broken"):
        await risk_nodes.neutral_turn_node(
            {"ticker": "ACN", "risk_turns": [_turn(0)]}   # neutral at index 1
        )


@pytest.mark.anyio
async def test_node_detects_a_double_applied_reducer(monkeypatch):
    _patch_port(monkeypatch)
    state = {
        "ticker": "ACN",
        "risk_turns": [_turn(0), _turn(1), _turn(1, persona="aggressive")],
    }

    with pytest.raises(RuntimeError, match=r"indices are \[0, 1, 1\]"):
        await risk_nodes.neutral_turn_node(state)


@pytest.mark.anyio
async def test_node_detects_a_missing_turn(monkeypatch):
    _patch_port(monkeypatch)
    state = {"ticker": "ACN", "risk_turns": [_turn(0), _turn(2, persona="conservative")]}

    with pytest.raises(RuntimeError, match="expected"):
        await risk_nodes.neutral_turn_node(state)


@pytest.mark.anyio
async def test_node_names_a_stale_checkpoint_rather_than_failing_later(monkeypatch):
    _patch_port(monkeypatch)
    state = {"ticker": "ACN", "risk_turns": [{"turn_index": 0}]}

    with pytest.raises(TypeError, match="predates a risk schema change"):
        await risk_nodes.neutral_turn_node(state)


# ---------------------------------------------------------------------------
# risk_close
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_close_node_records_the_cap():
    state = {"ticker": "ACN", "risk_turns": [_turn(i) for i in range(RISK_MAX_TURNS)]}

    assert await risk_nodes.risk_close_node(state) == {"risk_terminated_by": "round_cap"}


@pytest.mark.anyio
async def test_close_node_records_a_skipped_panel():
    assert await risk_nodes.risk_close_node({"ticker": "ACN"}) == {
        "risk_terminated_by": "no_debate"
    }


@pytest.mark.anyio
async def test_close_node_refuses_to_close_a_live_panel():
    state = {
        "ticker": "ACN",
        "risk_turns": [_turn(0)],
        "debate_turns": [object()],
    }

    with pytest.raises(RuntimeError, match="conditional edge map is wrong"):
        await risk_nodes.risk_close_node(state)
