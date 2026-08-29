"""Run-level cost and deadline accounting.

`RunBudget` and `cost_events` exist to answer one question a pure router can
check without I/O or a clock argument owned by the router itself: "has this
run already spent (or outlived) what it was allowed to?" Same shape as
`as_of_date` — set once at the CLI boundary, never computed or mutated
inside a node, so a breach test is deterministic (Phase 6 §0 established
this rule for dates; it applies just as much to money and to wall-clock
deadlines).

MAX_ROUNDS (debate_router.py / risk_router.py) is deliberately NOT here and
this module does not touch those routers. Budget/deadline are a *second,
independent* run-level abort layer, wrapped around the existing debate/risk
routers in graph.py rather than merged into them — merging them would let a
run-level abort masquerade as a debate-quality outcome, which is exactly
what must not happen (see graph.py's `guarded()`).
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RunBudget(BaseModel):
    """Set once per run, at the CLI boundary. Never mutated."""

    max_usd: float
    deadline_utc: datetime  # start_time + wall_clock_timeout, computed once


class CostEvent(BaseModel):
    """One LLM call's cost, as logged. Lands in `TradingState.cost_events`
    (an `operator.add` channel, same reducer as `debate_turns`/`risk_turns`)
    so a pure router can see cumulative run spend without re-reading the
    cost-log file.

    `event_id` is what makes resume idempotent: `total_spend()` dedupes on
    it rather than trusting that the `operator.add` reducer never double
    appends a pending write after a crash (the same open question Phase 5
    left for `debate_turns`, now with money attached — see
    trading_state.py's comment on that channel).
    """

    event_id: str
    node: str
    model: str  # the model actually used, from the response object
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    usd: float


class RunTermination(str, Enum):
    COMPLETED = "completed"
    BUDGET_EXCEEDED = "budget_exceeded"
    DEADLINE_EXCEEDED = "deadline_exceeded"


def total_spend(events: list[CostEvent]) -> float:
    """Sum of `usd` over `events`, counting each `event_id` once.

    Dedup lives here, in the reader, not in the `operator.add` reducer —
    the reducer stays the simple, well-understood append-only channel every
    other cyclic-node list in this state uses. If a crash-and-resume ever
    does append the same event twice, `total_spend` still reports the right
    total; the duplicate itself is surfaced as a warning so it is visible
    evidence rather than a silently-corrected number.
    """
    seen: set[str] = set()
    total = 0.0
    for event in events:
        if event.event_id in seen:
            logger.warning(
                "duplicate cost event %s for node %s — resume likely "
                "re-appended a pending write; counted once",
                event.event_id,
                event.node,
            )
            continue
        seen.add(event.event_id)
        total += event.usd
    return total
