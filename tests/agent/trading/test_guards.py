"""check_run_guards — pure, clock-injected, evaluated at zero API cost.

Same posture as test_debate_router.py: a router that is a pure function of
its inputs can be checked over every reachable case directly, rather than
inferred from a handful of live runs. The live budget/deadline breach runs
(criteria 5/6) then verify integration — that a real breach actually routes
to graceful_abort_node — which is a strictly weaker claim than this file
proves.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.agent.trading.application.guards import check_run_guards
from app.agent.trading.domain.budget import CostEvent, RunBudget, RunTermination

NOW = datetime(2026, 8, 26, 12, 0, 0, tzinfo=timezone.utc)


def _event(usd: float, event_id: str = "e1") -> CostEvent:
    return CostEvent(
        event_id=event_id,
        node="bull_turn",
        model="claude-haiku-4-5-20251001",
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        usd=usd,
    )


def _budget(max_usd: float = 0.75, deadline: datetime = NOW + timedelta(hours=1)) -> RunBudget:
    return RunBudget(max_usd=max_usd, deadline_utc=deadline)


def test_clean_run_returns_none():
    assert check_run_guards([_event(0.10)], _budget(), NOW) is None


def test_no_events_returns_none():
    assert check_run_guards([], _budget(), NOW) is None


def test_budget_exceeded_at_the_boundary():
    """>= , not > — a run at exactly the cap is over it, not still clean."""
    events = [_event(0.75)]
    assert check_run_guards(events, _budget(max_usd=0.75), NOW) == RunTermination.BUDGET_EXCEEDED


def test_budget_just_under_is_clean():
    events = [_event(0.7499)]
    assert check_run_guards(events, _budget(max_usd=0.75), NOW) is None


def test_budget_exceeded_across_multiple_events():
    events = [_event(0.30, "e1"), _event(0.30, "e2"), _event(0.30, "e3")]
    assert check_run_guards(events, _budget(max_usd=0.75), NOW) == RunTermination.BUDGET_EXCEEDED


def test_deadline_exceeded_at_the_boundary():
    budget = _budget(deadline=NOW)
    assert check_run_guards([], budget, NOW) == RunTermination.DEADLINE_EXCEEDED


def test_deadline_just_before_is_clean():
    budget = _budget(deadline=NOW + timedelta(seconds=1))
    assert check_run_guards([], budget, NOW) is None


def test_deadline_exceeded_even_with_zero_spend():
    """The two guards are independent — a hung run with no cost yet must
    still be catchable by the deadline alone."""
    budget = _budget(max_usd=1000.0, deadline=NOW - timedelta(seconds=1))
    assert check_run_guards([], budget, NOW) == RunTermination.DEADLINE_EXCEEDED


def test_budget_checked_before_deadline():
    """Both breached at once: budget is checked first (arbitrary but fixed
    ordering) — asserted so the precedence is a documented behavior, not an
    accident of implementation order that a future edit could silently flip."""
    budget = _budget(max_usd=0.10, deadline=NOW - timedelta(seconds=1))
    assert check_run_guards([_event(0.50)], budget, NOW) == RunTermination.BUDGET_EXCEEDED


def test_duplicate_event_id_does_not_double_count_toward_budget():
    """The resume-dedup path (total_spend) must hold under the guard too —
    a duplicated pending write must not make a clean run look like a false
    breach."""
    events = [_event(0.40, "e1"), _event(0.40, "e1")]  # same event_id twice
    assert check_run_guards(events, _budget(max_usd=0.75), NOW) is None
