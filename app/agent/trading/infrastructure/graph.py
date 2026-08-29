from datetime import datetime, timezone

from langgraph.graph import StateGraph, START, END

from app.agent.trading.application.debate_nodes import (
    bear_turn_node,
    bull_turn_node,
    debate_close_node,
)
from app.agent.trading.application.debate_router import next_debate_step
from app.agent.trading.application.guards import check_run_guards
from app.agent.trading.application.nodes import (
    fundamentals_node,
    graceful_abort_node,
    technical_node,
    news_node,
    sentiment_node,
    synthesizer_node,
)
from app.agent.trading.application.risk_nodes import (
    aggressive_turn_node,
    conservative_turn_node,
    neutral_turn_node,
    risk_close_node,
)
from app.agent.trading.application.risk_router import next_risk_step
from app.agent.trading.domain.trading_state import TradingState


def _guarded(inner):
    """Wraps a router (or a bare next-node name, for what used to be a plain
    edge) with the run-level budget/deadline check, run BEFORE the wrapped
    router ever sees the state. Every LLM-calling node's incoming edge goes
    through this — see build_trading_graph below — so a breach is reachable
    from anywhere in the graph, independent of what debate_router.py/
    risk_router.py would otherwise decide (see application/guards.py's
    module docstring for why those stay untouched).

    `budget` missing from state (every graph-level test invokes the graph
    with just {ticker, as_of_date}, no budget) means this run opted out of
    the guard entirely — not a breach. Only cli.py's real invocations set a
    RunBudget, and only those are ever gated.
    """
    route = inner if callable(inner) else (lambda state: inner)

    def router(state):
        budget = state.get("budget")
        if budget is not None:
            events = state.get("cost_events") or []
            if check_run_guards(events, budget, datetime.now(timezone.utc)) is not None:
                return "abort"
        return route(state)

    return router

# One entry per analyst as the CLI exposes it; the value is that analyst's
# node chain in run order. "news" is two nodes because the deterministic
# sentiment aggregation is part of the same analyst — it reads news_digest
# and nothing else, so it is never independently selectable.
ANALYST_CHAINS = {
    "fundamentals": (("fundamentals", fundamentals_node),),
    "technical": (("technical", technical_node),),
    "news": (("news", news_node), ("sentiment", sentiment_node)),
}
ALL_ANALYSTS = tuple(ANALYST_CHAINS)

# Two cycles, not one. The debate is bull/bear alternating under
# next_debate_step; the risk panel is neutral/aggressive/conservative
# rotating under next_risk_step, entered only after the debate closes. Each
# is its own conditional-edge group because a cycle cannot be expressed as a
# zip(chain, chain[1:]) edge pair — this is the same reason DEBATE_NODES
# grew a second shape rather than one more tuple entry in Phase 5, and
# RISK_NODES repeats it for a three-way rotation instead of two-way
# alternation.
DEBATE_NODES = (("bull_turn", bull_turn_node), ("bear_turn", bear_turn_node))
RISK_NODES = (
    ("neutral_turn", neutral_turn_node),
    ("aggressive_turn", aggressive_turn_node),
    ("conservative_turn", conservative_turn_node),
)

# debate_close is the single point every exit path from the debate cycle
# passes through (cap or no-evidence), which is where the termination
# reason gets recorded and where the risk panel's conditional ENTRY edge
# attaches — same reason the analyst->debate entry edge is conditional
# rather than plain: a plain edge into neutral_turn would run one turn
# before any router saw the state, so risk_router's no-debate short-circuit
# could never skip the panel.
POST_DEBATE_NODES = (("debate_close", debate_close_node),)

# risk_close plays the identical role for the risk cycle that debate_close
# plays for the debate: it is where risk_terminated_by gets recorded, and
# every exit path from RISK_NODES routes through it before the synthesizer
# — the memo is the pipeline's output contract, so a partial run must still
# say what it did and did not see rather than skip straight to a stub.
POST_RISK_NODES = (("risk_close", risk_close_node), ("synthesizer", synthesizer_node))


def build_trading_graph(checkpointer, interrupt_after=None, analysts=None):
    """`interrupt_after` takes a list of node names to stop after, e.g.
    ["technical"]. Used by the checkpoint round-trip test to stop the graph
    deterministically at a node boundary.

    `interrupt_after` on either CYCLIC node group interrupts after EVERY
    execution of a member — `interrupt_after=["bull_turn"]` stops after turn
    0, then turn 2, then turn 4; `interrupt_after=["neutral_turn"]` stops
    after risk turn 0, then turn 3. Read N interrupts as N interrupts, not a
    runaway (same note Phase 5 left for DEBATE_NODES).

    Existing pre-Phase-6 threads (including anything checkpointed against
    the Phase 5 graph, which had `risk` as a single stub node and no risk
    cycle) are unresumable under this graph shape — the node names differ
    entirely. Use a fresh --thread-id.

    `analysts` selects which analyst legs run, e.g. ["news"]; None runs all
    of them. Selection order is ignored — legs always run in
    ANALYST_CHAINS order so that a subset run is a strict subsequence of
    the full run, and any later cross-analyst dependency holds in both."""
    unknown = sorted(set(analysts or ()) - set(ALL_ANALYSTS))
    if unknown:
        raise ValueError(
            f"unknown analyst(s): {', '.join(unknown)} — "
            f"choose from {', '.join(ALL_ANALYSTS)}"
        )
    selected = (
        ALL_ANALYSTS
        if analysts is None
        else tuple(a for a in ALL_ANALYSTS if a in set(analysts))
    )
    if not selected:
        raise ValueError("at least one analyst must be selected")

    builder = StateGraph(TradingState)

    analyst_chain = [node for a in selected for node in ANALYST_CHAINS[a]]
    debate_tail = list(POST_DEBATE_NODES)
    risk_tail = list(POST_RISK_NODES)

    for name, fn in (
        analyst_chain
        + list(DEBATE_NODES)
        + debate_tail
        + list(RISK_NODES)
        + risk_tail
    ):
        builder.add_node(name, fn)
    builder.add_node("graceful_abort", graceful_abort_node)

    # Phase 8: every edge into an LLM-calling node goes through _guarded(),
    # which checks the run-level budget/deadline BEFORE the wrapped router
    # ever runs and routes to "abort" on a breach — reachable from anywhere
    # in the graph, independent of debate/risk-quality routing (see
    # application/guards.py). The very first edge (START -> first analyst)
    # is deliberately left plain: at t=0 with zero spend nothing can have
    # breached yet, so guarding it would only add a router call that can
    # never return "abort".
    builder.add_edge(START, analyst_chain[0][0])
    for (prev, _), (nxt, _) in zip(analyst_chain, analyst_chain[1:]):
        builder.add_conditional_edges(
            prev, _guarded(nxt), {nxt: nxt, "abort": "graceful_abort"}
        )

    # The debate cycle's entry edge is conditional, same reasoning as Phase
    # 5: the router must see the state before the first turn runs, or the
    # no-evidence skip can never fire. Both debate nodes get the full route
    # map including the never-taken same-side branch — if it ever fires,
    # the result is a visible loop the alternation assert in debate_nodes
    # then names precisely, not a KeyError deep in LangGraph routing.
    debate_route_map = {
        "bull": "bull_turn", "bear": "bear_turn", "done": debate_tail[0][0],
        "abort": "graceful_abort",
    }
    for src in (analyst_chain[-1][0], "bull_turn", "bear_turn"):
        builder.add_conditional_edges(src, _guarded(next_debate_step), debate_route_map)

    for (prev, _), (nxt, _) in zip(debate_tail, debate_tail[1:]):
        builder.add_edge(prev, nxt)

    # Same shape one cycle up: debate_close is the risk panel's conditional
    # entry point, and every risk node carries the full three-way route map
    # plus "done".
    risk_route_map = {
        "neutral": "neutral_turn",
        "aggressive": "aggressive_turn",
        "conservative": "conservative_turn",
        "done": risk_tail[0][0],
        "abort": "graceful_abort",
    }
    for src in (debate_tail[-1][0], "neutral_turn", "aggressive_turn", "conservative_turn"):
        builder.add_conditional_edges(src, _guarded(next_risk_step), risk_route_map)

    for (prev, _), (nxt, _) in zip(risk_tail, risk_tail[1:]):
        # risk_close -> synthesizer today (risk_tail has exactly two
        # entries) — synthesizer is an LLM-calling node, so this edge is
        # guarded like every other one above.
        builder.add_conditional_edges(
            prev, _guarded(nxt), {nxt: nxt, "abort": "graceful_abort"}
        )
    builder.add_edge(risk_tail[-1][0], END)
    builder.add_edge("graceful_abort", END)

    return builder.compile(
        checkpointer=checkpointer, interrupt_after=interrupt_after or []
    )
