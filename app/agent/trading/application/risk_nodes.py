"""One node per risk-panel TURN, cycled by a conditional edge — same reason
as debate_nodes.py: one node execution is one LangGraph super-step, so a
kill mid-panel resumes at the exact turn in flight rather than losing a
whole round.
"""

from app.agent.trading.application.risk_router import RISK_MAX_TURNS, next_risk_step
from app.agent.trading.domain.risk import PERSONAS, Persona, RiskTurn
from app.agent.trading.domain.trading_state import TradingState
from app.agent.trading.infrastructure.risk_port import run_risk_turn


async def _risk_turn(state: TradingState, persona: Persona) -> dict:
    turns = state.get("risk_turns") or []

    # Same reason as debate_nodes._debate_turn: a checkpoint predating a
    # risk schema change deserializes as plain dicts with nothing raising at
    # the read, three levels deep here (RiskTurn -> RiskTurnPayload ->
    # RiskFactor/RiskScore).
    stale = [t for t in turns if not isinstance(t, RiskTurn)]
    if stale:
        raise TypeError(
            f"risk_turns holds {len(stale)} {type(stale[0]).__name__} "
            f"entr{'y' if len(stale) == 1 else 'ies'} instead of RiskTurn — this "
            f"checkpoint predates a risk schema change and cannot be resumed. "
            f"Re-run the ticker under a new --thread-id."
        )

    turn_index = len(turns)

    # Layer 3 — defensive. If the router is correct none of these fire.
    if turn_index >= RISK_MAX_TURNS:
        raise RuntimeError(
            f"{persona}_turn entered at turn_index={turn_index} with "
            f"RISK_MAX_TURNS={RISK_MAX_TURNS} — router bypassed. Refusing to run."
        )

    indices = [t.turn_index for t in turns]
    if indices != list(range(turn_index)):
        raise RuntimeError(
            f"risk_turns indices are {indices}, expected "
            f"{list(range(turn_index))} — the add-reducer double-applied on "
            f"resume, or a turn was lost. Refusing to append turn "
            f"{turn_index} onto a transcript with gaps or duplicates."
        )

    expected = PERSONAS[turn_index % len(PERSONAS)]
    if persona != expected:
        raise RuntimeError(
            f"Rotation broken: {persona} ran at turn_index={turn_index}, "
            f"expected {expected}."
        )

    turn = await run_risk_turn(state=state, persona=persona, turn_index=turn_index)
    print(
        f"[risk] {persona} turn {turn_index} (round {turn.round_num}) "
        f"proposes={len(turn.payload.proposes)} scores={len(turn.payload.scores)} "
        f"flags={len(turn.guard_flags)}"
    )

    return {"risk_turns": [turn]}


async def neutral_turn_node(state: TradingState) -> dict:
    return await _risk_turn(state, "neutral")


async def aggressive_turn_node(state: TradingState) -> dict:
    return await _risk_turn(state, "aggressive")


async def conservative_turn_node(state: TradingState) -> dict:
    return await _risk_turn(state, "conservative")


async def risk_close_node(state: TradingState) -> dict:
    """Records WHY the panel stopped — the single place every exit path
    (cap, no-debate) passes through, same role as debate_close_node."""
    step = next_risk_step(state)
    if step != "done":
        raise RuntimeError(
            f"risk_close entered while the router still wants a {step} turn "
            f"at turn_index={len(state.get('risk_turns') or [])} — the "
            f"conditional edge map is wrong."
        )
    turns = state.get("risk_turns") or []
    reason = "round_cap" if len(turns) >= RISK_MAX_TURNS else "no_debate"
    print(f"[risk] closed after {len(turns)} turn(s): {reason}")
    return {"risk_terminated_by": reason}
