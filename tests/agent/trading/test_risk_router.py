"""The termination proof for the risk panel, mirroring
test_debate_router.py: `next_risk_step` is a pure function of state, so it
can be evaluated over every reachable input in milliseconds at zero API cost.
This is what closes Phase 6 exit criterion 1 — no live run required.
"""

from __future__ import annotations

import pytest

from app.agent.trading.application.risk_router import (
    PERSONAS,
    RISK_MAX_ROUNDS,
    RISK_MAX_TURNS,
    next_risk_step,
)
from app.agent.trading.domain.debate import DebateClaim, DebateTurn, DebateTurnPayload
from app.agent.trading.domain.risk import RiskTurn, RiskTurnPayload


def _debate_turn(index: int) -> DebateTurn:
    return DebateTurn(
        turn_index=index,
        round_num=(index // 2) + 1,
        side="bull" if index % 2 == 0 else "bear",
        payload=DebateTurnPayload(
            stance="hold",
            argument="stub",
            claims=[DebateClaim(claim_id=f"c{index}", text="t", evidence_ref="none")],
        ),
    )


def _risk_turn(index: int) -> RiskTurn:
    return RiskTurn(
        turn_index=index,
        round_num=(index // len(PERSONAS)) + 1,
        persona=PERSONAS[index % len(PERSONAS)],
        payload=RiskTurnPayload(argument="stub"),
    )


def _state(n: int, with_debate: bool = True, **extra) -> dict:
    state = {"risk_turns": [_risk_turn(i) for i in range(n)]}
    if with_debate:
        state["debate_turns"] = [_debate_turn(0)]
    state.update(extra)
    return state


# ---------------------------------------------------------------------------
# Exhaustive: every reachable turn count, plus a margin past the cap, crossed
# with whether a debate happened at all.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", range(0, RISK_MAX_TURNS + 5))
@pytest.mark.parametrize("with_debate", [True, False])
def test_router_is_total_and_alternates_below_the_cap(n, with_debate):
    step = next_risk_step(_state(n, with_debate=with_debate))
    if n >= RISK_MAX_TURNS:
        assert step == "done"
    elif n == 0 and not with_debate:
        assert step == "done"
    else:
        assert step == PERSONAS[n % len(PERSONAS)]
    assert step is not None


def test_max_turns_is_derived_from_max_rounds_and_persona_count():
    """Guards the recursion_limit the CLI derives from this."""
    assert RISK_MAX_TURNS == len(PERSONAS) * RISK_MAX_ROUNDS
    assert RISK_MAX_ROUNDS == 2
    assert RISK_MAX_TURNS == 6


def test_router_reaches_the_cap_regardless_of_turn_content():
    """No productivity/convergence clause exists here at all — unlike Phase
    5's now-removed UNPRODUCTIVE_STOP, there was never one to remove. The cap
    is the only lever, full stop."""
    assert next_risk_step(_state(RISK_MAX_TURNS)) == "done"


# ---------------------------------------------------------------------------
# No-debate entry guard
# ---------------------------------------------------------------------------

def test_router_skips_the_panel_when_no_debate_took_place():
    assert next_risk_step({}) == "done"
    assert next_risk_step({"risk_turns": []}) == "done"
    assert next_risk_step({"risk_turns": [], "debate_turns": []}) == "done"


def test_router_opens_with_neutral_when_a_debate_happened():
    assert next_risk_step({"debate_turns": [_debate_turn(0)]}) == "neutral"


def test_no_debate_check_only_applies_on_the_opening_turn():
    """Once turns exist the debate question is settled — re-asking it mid-
    panel would end a live panel the moment debate_turns read empty under a
    different state shape, mirroring the debate router's own guard."""
    state = {"risk_turns": [_risk_turn(0)]}   # no debate_turns key at all
    assert next_risk_step(state) == "aggressive"


# ---------------------------------------------------------------------------
# Rotation order
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n,expected",
    [(0, "neutral"), (1, "aggressive"), (2, "conservative"),
     (3, "neutral"), (4, "aggressive"), (5, "conservative")],
)
def test_persona_rotation_order(n, expected):
    assert next_risk_step(_state(n)) == expected
