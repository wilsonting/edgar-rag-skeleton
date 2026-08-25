"""The termination proof.

The stated exit criterion — "across 5 test runs, the debate always terminates
within the round cap" — cannot establish a bound. Five clean stochastic runs
are consistent with a 10% runaway rate at roughly 59% probability; that is a
smoke test wearing the clothes of a guarantee.

`next_debate_step` is a pure function of state, so it can be evaluated over
every reachable input in milliseconds at zero API cost. THIS is the test that
closes the termination criterion. The live runs then verify *integration* —
that the router is actually wired to the edges — which is a strictly weaker
claim and worth writing down as such.
"""

from __future__ import annotations

import pytest

from app.agent.trading.application.debate_router import (
    MAX_ROUNDS,
    MAX_TURNS,
    next_debate_step,
    termination_reason,
)
from app.agent.trading.domain.debate import DebateClaim, DebateTurn, DebateTurnPayload


def _stub_turn(index: int, productive: bool = True) -> DebateTurn:
    return DebateTurn(
        turn_index=index,
        round_num=(index // 2) + 1,
        side="bull" if index % 2 == 0 else "bear",
        payload=DebateTurnPayload(
            stance="hold",
            argument="stub",
            claims=[
                DebateClaim(
                    claim_id=f"c{index}" if productive else "c0",
                    text="stub",
                    evidence_ref="none",
                )
            ],
        ),
        productive=productive,
    )


def _state(n: int, productive: bool = True, **extra) -> dict:
    state = {"debate_turns": [_stub_turn(i, productive) for i in range(n)]}
    # evidence present unless a test says otherwise, so the no-evidence
    # short-circuit doesn't quietly answer questions about the cap
    state.setdefault("fundamentals_report", object())
    state.update(extra)
    return state


# ---------------------------------------------------------------------------
# Exhaustive: every reachable turn count, plus five past the cap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", range(0, MAX_TURNS + 5))
def test_router_alternates_below_the_cap_and_stops_at_or_above_it(n):
    step = next_debate_step(_state(n))
    if n >= MAX_TURNS:
        assert step == "done"
    else:
        assert step == ("bull" if n % 2 == 0 else "bear")


def test_router_reaches_the_cap_even_when_every_turn_introduced_a_new_claim():
    """The cap wins because it's the only lever left, not because it's
    racing a productivity clause — that clause was removed 2026-08-24 (see
    the module docstring): it never fired across any measured debate, so
    MAX_TURNS is the sole termination bound now."""
    assert next_debate_step(_state(MAX_TURNS, productive=True)) == "done"


def test_max_turns_is_derived_from_max_rounds():
    """Guards the recursion_limit the CLI derives from MAX_ROUNDS."""
    assert MAX_TURNS == 2 * MAX_ROUNDS


# ---------------------------------------------------------------------------
# Regression: the router must not resurrect a productivity-based early stop.
# `productive` is still computed and still recorded on DebateTurn (§ known
# gaps), but it must have ZERO effect on routing — these turns are all
# `productive=False` and the router must still alternate through them to the
# cap rather than stopping early, or the dead branch is back.
# ---------------------------------------------------------------------------

def test_the_router_alternates_through_unproductive_turns_to_the_cap():
    turns = [_stub_turn(i, productive=False) for i in range(MAX_TURNS - 1)]
    state = {"debate_turns": turns, "fundamentals_report": object()}
    assert next_debate_step(state) == (
        "bull" if len(turns) % 2 == 0 else "bear"
    )


def test_an_all_unproductive_transcript_still_needs_the_cap_to_stop():
    for n in range(MAX_TURNS):
        state = _state(n, productive=False)
        assert next_debate_step(state) != "done"
    assert next_debate_step(_state(MAX_TURNS, productive=False)) == "done"


def test_router_skips_the_debate_when_no_analyst_ran():
    """A --only run that excluded every analyst leg would otherwise produce a
    debate over an empty pack: two models arguing from nothing, which reads
    like a debate and is theatre."""
    assert next_debate_step({}) == "done"
    assert next_debate_step({"debate_turns": []}) == "done"


@pytest.mark.parametrize(
    "key", ["fundamentals_report", "technical_report", "news_digest"]
)
def test_router_opens_with_bull_when_any_analyst_output_is_present(key):
    assert next_debate_step({key: object()}) == "bull"


def test_no_evidence_check_only_applies_on_the_opening_turn():
    """Once turns exist the pack question is settled; re-asking it mid-debate
    would end a live debate the moment a report was read out of state under a
    different key."""
    state = {"debate_turns": [_stub_turn(0)]}   # no analyst outputs at all
    assert next_debate_step(state) == "bear"


# ---------------------------------------------------------------------------
# termination_reason
# ---------------------------------------------------------------------------

def test_termination_reason_reports_the_cap():
    assert termination_reason(_state(MAX_TURNS)) == "round_cap"


def test_termination_reason_reports_no_evidence():
    assert termination_reason({}) == "no_evidence"


def test_termination_reason_has_no_third_outcome():
    """"unproductive" was a reachable termination_reason before 2026-08-24.
    It no longer is, because the router clause that produced it is gone —
    the only inputs where next_debate_step returns "done" are the cap and
    the empty-evidence case, so those are the only two reasons left."""
    assert termination_reason(_state(MAX_TURNS, productive=False)) == "round_cap"


def test_every_done_state_has_a_reason():
    """No exit path leaves debate_terminated_by empty — an unrecorded reason
    would read in the memo as a debate that simply ended."""
    states = [{}, _state(MAX_TURNS), _state(MAX_TURNS, productive=False)]
    for state in states:
        assert next_debate_step(state) == "done"
        assert termination_reason(state) in {"round_cap", "no_evidence"}
