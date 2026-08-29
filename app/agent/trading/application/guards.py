"""Run-level abort guard — budget and wall-clock deadline.

Deliberately separate from debate_router.py/risk_router.py and does not
modify either: MAX_ROUNDS/RISK_MAX_ROUNDS are debate/risk-QUALITY
termination levers (see those modules' own docstrings for why each is the
ONLY one), while a budget or deadline breach is a run-level abort that must
be reachable from ANYWHERE in the graph, independent of what the debate or
risk panel would otherwise decide. Mixing the two would let a run-level
abort masquerade as a debate/risk-quality outcome — the two concerns stay in
different modules on purpose, wired together only in graph.py's `guarded()`.

Pure and clock-injected, same reasoning debate_router.py gives for being a
pure function of state: a router that can be evaluated over every reachable
input in milliseconds at zero API cost is a proof of termination, not a
sample of it. `now` is a parameter rather than `datetime.now()` inside the
function for exactly that reason — the budget/deadline breach tests
(criteria 5 and 6) need to construct an exact boundary case deterministically,
which a router that reads the wall clock itself could not do.
"""

from __future__ import annotations

from datetime import datetime

from app.agent.trading.domain.budget import CostEvent, RunBudget, RunTermination, total_spend


def check_run_guards(
    events: list[CostEvent], budget: RunBudget, now: datetime
) -> RunTermination | None:
    """Checked before every LLM-calling node (see graph.py's `guarded()`).

    Ordering means a run can overshoot `max_usd` by at most one call's
    cost — the check runs BEFORE the next call, not after, so it cannot see
    that call's spend coming. Pre-call token estimation would close that gap
    only by guessing, which is a worse failure mode than a small, bounded
    overshoot: `max_usd` is set below any real pain threshold precisely to
    absorb it. This is the direct fix for the failure mode a ledger checked
    on every edge is for — unbounded iteration with no budget at all — not
    an attempt to make the overshoot exactly zero.

    A second, smaller overshoot source, live-verified (Phase 8 criterion 7,
    crash-resume test): `events` here is `state["cost_events"]`, which can
    UNDER-count real spend when a node's LLM call is billed (its cost-log
    line already written to disk) but the node itself then crashes or is
    retried before that call's CostEvent reaches state — the retry bills
    again for real, and this guard never sees the discarded first attempt's
    cost at all. `cost_log.log_run_summary` reconciles the RECORDED total
    against disk after the fact (see `cost_ledger_gap_usd`), but this
    router runs mid-flight and stays a pure function of state on purpose —
    it does not read the disk log — so it inherits the same blind spot.
    Bounded by how many node-level retries a run experiences, which in
    practice is rare and small per retry, not by design.
    """
    if total_spend(events) >= budget.max_usd:
        return RunTermination.BUDGET_EXCEEDED
    if now >= budget.deadline_utc:
        return RunTermination.DEADLINE_EXCEEDED
    return None
