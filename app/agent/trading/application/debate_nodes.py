"""One node per debate TURN, cycled by a conditional edge.

This is the decision the phase turns on. LangGraph persists a checkpoint at
the end of every super-step, and in a sequential graph one node execution is
one super-step:

    for r in range(rounds) inside a single debate node -> 1 checkpoint, at
        node exit. A kill mid-round-2 resumes at turn 0 and the whole debate
        is lost.
    one node per turn, cycled                          -> 2 x rounds
        checkpoints. A kill resumes at the exact turn in flight.

"Resumes from the last per-round checkpoint, not from the start" is
unsatisfiable under the loop design — there is no flag that adds intra-node
checkpointing. Everything else about this phase (the conditional edges in
graph.py, the add-reducer on debate_turns) is downstream of that.
"""

from app.agent.trading.application.debate_router import (
    MAX_TURNS,
    next_debate_step,
    termination_reason,
)
from app.agent.trading.domain.debate import DebateTurn, Side
from app.agent.trading.domain.trading_state import TradingState
from app.agent.trading.infrastructure.debate_port import run_debate_turn


async def _debate_turn(state: TradingState, side: Side) -> dict:
    turns = state.get("debate_turns") or []

    # The same failure mode sentiment_node guards against, one level deeper:
    # DebateTurn nests DebateTurnPayload nests DebateClaim, so a checkpoint
    # predating a schema change deserializes as plain dicts with nothing
    # raising at the read. Name it here rather than as an AttributeError
    # three frames on.
    stale = [t for t in turns if not isinstance(t, DebateTurn)]
    if stale:
        raise TypeError(
            f"debate_turns holds {len(stale)} {type(stale[0]).__name__} "
            f"entr{'y' if len(stale) == 1 else 'ies'} instead of DebateTurn — this "
            f"checkpoint predates a debate schema change and cannot be resumed. "
            f"Re-run the ticker under a new --thread-id."
        )

    turn_index = len(turns)

    # Layer 3 of the termination guarantee — defensive. If the router is
    # correct none of these ever fire; each of them names a specific way the
    # wiring could be wrong, which is worth more than a KeyError deep in
    # LangGraph routing that reads as a framework bug.
    if turn_index >= MAX_TURNS:
        raise RuntimeError(
            f"{side}_turn entered at turn_index={turn_index} with "
            f"MAX_TURNS={MAX_TURNS} — router bypassed. Refusing to run."
        )

    # Contiguity, not merely "index not already present". A pending write
    # re-applied on resume appends a DUPLICATE of the last turn, leaving
    # [0,1,2,2] — whose length is 4, so an "is turn_index already taken?"
    # check would look clean and the run would carry on with a transcript one
    # turn longer than the cap allows. The invariant that actually holds is
    # that indices are exactly range(len(turns)); anything else means the
    # add-reducer double-applied and debate_turns needs a
    # dedup-on-turn_index reducer rather than plain operator.add.
    #
    # Two live crash-and-resume runs (2026-08-23) never tripped this, which
    # is the outcome that lets plain `add` stay. Keep it anyway: it costs a
    # list comparison per turn and it is the only thing that would catch the
    # double-apply loudly rather than as a transcript one turn too long.
    indices = [t.turn_index for t in turns]
    if indices != list(range(turn_index)):
        raise RuntimeError(
            f"debate_turns indices are {indices}, expected "
            f"{list(range(turn_index))} — the add-reducer double-applied on "
            f"resume, or a turn was lost. Refusing to append turn "
            f"{turn_index} onto a transcript with gaps or duplicates."
        )

    expected = "bull" if turn_index % 2 == 0 else "bear"
    if side != expected:
        raise RuntimeError(
            f"Alternation broken: {side} ran at turn_index={turn_index}, "
            f"expected {expected}."
        )

    turn = await run_debate_turn(state=state, side=side, turn_index=turn_index)
    print(
        f"[debate] {side} turn {turn_index} (round {turn.round_num}) "
        f"stance={turn.payload.stance} claims={len(turn.payload.claims)} "
        f"productive={turn.productive} flags={len(turn.guard_flags)}"
    )

    # A DELTA, never the accumulated list. `operator.add` concatenates, so
    # returning the whole list doubles it every super-step — and that failure
    # looks exactly like the runaway loop this phase exists to bound.
    return {"debate_turns": [turn]}


async def bull_turn_node(state: TradingState) -> dict:
    return await _debate_turn(state, "bull")


async def bear_turn_node(state: TradingState) -> dict:
    return await _debate_turn(state, "bear")


async def debate_close_node(state: TradingState) -> dict:
    """Records WHY the debate stopped, on the way out of the cycle.

    The router is a pure function and writes no state, so without this node
    nothing would persist the termination reason — and a capped debate reads
    in the memo exactly like a resolved one unless the memo says otherwise.
    It sits at the head of the post-debate chain, which is the single place
    every exit path (cap, no-evidence) passes through.

    Asserted rather than assumed: reaching this node while the router still
    wants another turn means the routing table is wrong.
    """
    step = next_debate_step(state)
    if step != "done":
        raise RuntimeError(
            f"debate_close entered while the router still wants a {step} turn "
            f"at turn_index={len(state.get('debate_turns') or [])} — the "
            f"conditional edge map is wrong."
        )
    reason = termination_reason(state)
    turns = state.get("debate_turns") or []
    print(f"[debate] closed after {len(turns)} turn(s): {reason}")
    return {"debate_terminated_by": reason}
