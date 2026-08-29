"""Node-level wiring: the layer-3 asserts, the delta shape, and the close node.

These are the checks that fire when the ROUTER is bypassed — a wiring bug the
router itself cannot see. If they ever trip in a live run the graph is wrong,
not the model.
"""

from __future__ import annotations

import pytest

import app.agent.trading.application.debate_nodes as debate_nodes
from app.agent.trading.application.debate_router import MAX_TURNS
from app.agent.trading.domain.debate import DebateClaim, DebateTurn, DebateTurnPayload


def _turn(index: int, side: str | None = None, productive: bool = True) -> DebateTurn:
    return DebateTurn(
        turn_index=index,
        round_num=(index // 2) + 1,
        side=side or ("bull" if index % 2 == 0 else "bear"),
        payload=DebateTurnPayload(
            stance="hold",
            argument="stub",
            claims=[DebateClaim(claim_id=f"c{index}", text="t", evidence_ref="none")],
        ),
        productive=productive,
    )


def _patch_port(monkeypatch, calls: list | None = None):
    async def fake_run(state, side, turn_index):
        if calls is not None:
            calls.append((side, turn_index))
        return _turn(turn_index, side)

    monkeypatch.setattr(debate_nodes, "run_debate_turn", fake_run)


@pytest.mark.anyio
async def test_bull_turn_returns_a_single_element_delta(monkeypatch):
    """Nodes return a DELTA. `operator.add` concatenates, so returning the
    accumulated list would double it every super-step — and that failure looks
    exactly like the runaway loop the round cap exists to bound."""
    _patch_port(monkeypatch)

    update = await debate_nodes.bull_turn_node({"ticker": "ACN", "debate_turns": [_turn(0), _turn(1)]})

    assert set(update) == {"debate_turns", "cost_events"}
    assert len(update["debate_turns"]) == 1
    assert update["debate_turns"][0].turn_index == 2


@pytest.mark.anyio
async def test_turn_index_and_round_come_from_the_transcript_length(monkeypatch):
    calls: list = []
    _patch_port(monkeypatch, calls)

    await debate_nodes.bull_turn_node({"ticker": "ACN"})
    await debate_nodes.bear_turn_node(
        {"ticker": "ACN", "debate_turns": [_turn(0)]}
    )

    assert calls == [("bull", 0), ("bear", 1)]


@pytest.mark.anyio
async def test_node_refuses_to_run_past_the_cap(monkeypatch):
    """Layer 3. If the router is correct this never fires; when it does, it
    names the bypass rather than surfacing as a mysterious extra turn."""
    _patch_port(monkeypatch)
    state = {"ticker": "ACN", "debate_turns": [_turn(i) for i in range(MAX_TURNS)]}

    with pytest.raises(RuntimeError, match="router bypassed"):
        await debate_nodes.bull_turn_node(state)


@pytest.mark.anyio
async def test_node_refuses_to_break_alternation(monkeypatch):
    _patch_port(monkeypatch)

    with pytest.raises(RuntimeError, match="Alternation broken"):
        await debate_nodes.bear_turn_node({"ticker": "ACN"})   # bear at index 0

    with pytest.raises(RuntimeError, match="Alternation broken"):
        await debate_nodes.bull_turn_node(
            {"ticker": "ACN", "debate_turns": [_turn(0)]}       # bull at index 1
        )


@pytest.mark.anyio
async def test_node_detects_a_double_applied_reducer(monkeypatch):
    """A pending write re-applied on resume appends a duplicate of the LAST
    turn, so the transcript reads [0,1,1] and its length still looks
    plausible. Contiguity is the invariant that catches that; "is this index
    already taken?" would not, because the next index is free either way."""
    _patch_port(monkeypatch)
    state = {"ticker": "ACN", "debate_turns": [_turn(0), _turn(1), _turn(1, side="bear")]}

    with pytest.raises(RuntimeError, match=r"indices are \[0, 1, 1\]"):
        await debate_nodes.bull_turn_node(state)


@pytest.mark.anyio
async def test_node_detects_a_missing_turn(monkeypatch):
    """The other half of the same invariant: a gap means a turn was lost."""
    _patch_port(monkeypatch)
    state = {"ticker": "ACN", "debate_turns": [_turn(0), _turn(2, side="bear")]}

    with pytest.raises(RuntimeError, match="expected"):
        await debate_nodes.bull_turn_node(state)


@pytest.mark.anyio
async def test_node_names_a_stale_checkpoint_rather_than_failing_later(monkeypatch):
    """DebateTurn nests DebateTurnPayload nests DebateClaim, so a checkpoint
    predating a schema change comes back as plain dicts with nothing raising
    at the read."""
    _patch_port(monkeypatch)
    state = {"ticker": "ACN", "debate_turns": [{"turn_index": 0}]}

    with pytest.raises(TypeError, match="predates a debate schema change"):
        await debate_nodes.bull_turn_node(state)


# ---------------------------------------------------------------------------
# debate_close
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_close_node_records_the_cap():
    state = {"ticker": "ACN", "debate_turns": [_turn(i) for i in range(MAX_TURNS)]}

    assert await debate_nodes.debate_close_node(state) == {
        "debate_terminated_by": "round_cap"
    }


@pytest.mark.anyio
async def test_close_node_records_a_skipped_debate():
    assert await debate_nodes.debate_close_node({"ticker": "ACN"}) == {
        "debate_terminated_by": "no_evidence"
    }


@pytest.mark.anyio
async def test_close_node_does_not_treat_an_unproductive_pair_as_done():
    """Regression for the removed early stop (2026-08-24): two consecutive
    productive=False turns used to be a legitimate reason for the router to
    say 'done', and this node to record it. It is not any more — the router
    only considers the debate done at the cap or with no evidence at all, so
    close_node reached on a 2-turn unproductive transcript must refuse, the
    same way it refuses on any other live debate."""
    state = {
        "ticker": "ACN",
        "debate_turns": [_turn(0, productive=False), _turn(1, productive=False)],
        "fundamentals_report": object(),
    }

    with pytest.raises(RuntimeError, match="conditional edge map is wrong"):
        await debate_nodes.debate_close_node(state)


@pytest.mark.anyio
async def test_close_node_refuses_to_close_a_live_debate():
    """Reaching the close node while the router still wants another turn
    means the conditional edge map is wrong."""
    state = {"ticker": "ACN", "debate_turns": [_turn(0)], "fundamentals_report": object()}

    with pytest.raises(RuntimeError, match="conditional edge map is wrong"):
        await debate_nodes.debate_close_node(state)
