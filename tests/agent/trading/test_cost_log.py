"""log_run_summary's disk reconciliation — Phase 8 criterion 7 finding.

Live-verified (a real `DEBATE_CRASH_AT_TURN` crash + resume): a node's LLM
call can already be billed to disk (`log_cost` writes synchronously, before
the node returns) and then never reach `TradingState.cost_events` at all,
because the node crashed or LangGraph retried it before that CostEvent
committed. The retry bills again for real and gets its own `event_id` — so
the disk log ends up STRICTLY LARGER than the state-derived total, never a
duplicate of it. `log_run_summary` must report the larger, honest figure and
flag that it did, not silently trust `state["cost_events"]` alone.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from app.agent.trading.domain.budget import CostEvent, RunBudget, RunTermination
from app.agent.trading.infrastructure import cost_log


def _event(node: str, usd: float, event_id: str) -> CostEvent:
    return CostEvent(
        event_id=event_id,
        node=node,
        model="claude-haiku-4-5-20251001",
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        usd=usd,
    )


def _disk_line(run_id: str, event_id: str, usd: float) -> str:
    return json.dumps({
        "kind": "cost_event",
        "run_id": run_id,
        "event_id": event_id,
        "estimated_cost_usd": usd,
    })


def _budget() -> RunBudget:
    return RunBudget(max_usd=0.75, deadline_utc=datetime.now(timezone.utc))


@pytest.fixture(autouse=True)
def _isolated_cost_log(tmp_path, monkeypatch):
    path = tmp_path / "cost-log.jsonl"
    monkeypatch.setattr(cost_log, "_COST_LOG_PATH", path)
    return path


def _last_line(path) -> dict:
    return json.loads(path.read_text().strip().splitlines()[-1])


def test_no_gap_when_disk_matches_state(tmp_path):
    path = cost_log._COST_LOG_PATH
    path.write_text(
        _disk_line("run-1", "bull_turn:0:aaa", 0.01) + "\n"
        + _disk_line("run-1", "bear_turn:1:bbb", 0.02) + "\n"
    )
    events = [_event("bull_turn", 0.01, "bull_turn:0:aaa"), _event("bear_turn", 0.02, "bear_turn:1:bbb")]

    cost_log.log_run_summary(
        run_id="run-1", ticker="ACN", as_of_date=date(2026, 8, 26),
        events=events, budget=_budget(), terminated_by=RunTermination.COMPLETED,
        wall_clock_s=12.0,
    )

    summary = _last_line(path)
    assert summary["total_usd"] == pytest.approx(0.03)
    assert summary["cost_ledger_gap_usd"] == 0.0
    assert summary["n_events"] == 2


def test_gap_flagged_when_a_crashed_call_never_reached_state(caplog):
    """The exact live-verified shape: turn 2 crashed after billing $0.00743
    to disk, then the resumed retry billed again for real under a DIFFERENT
    event_id. `events` (the state ledger) only ever sees the retry."""
    path = cost_log._COST_LOG_PATH
    path.write_text(
        _disk_line("run-2", "bull_turn:0:aaa", 0.006) + "\n"
        + _disk_line("run-2", "bear_turn:1:bbb", 0.007) + "\n"
        + _disk_line("run-2", "bull_turn:2:crashed", 0.00743) + "\n"   # never committed
        + _disk_line("run-2", "bull_turn:2:retried", 0.00768) + "\n"  # the successful retry
    )
    events = [
        _event("bull_turn", 0.006, "bull_turn:0:aaa"),
        _event("bear_turn", 0.007, "bear_turn:1:bbb"),
        _event("bull_turn", 0.00768, "bull_turn:2:retried"),
    ]

    with caplog.at_level("WARNING"):
        cost_log.log_run_summary(
            run_id="run-2", ticker="AVGO", as_of_date=date(2026, 8, 26),
            events=events, budget=_budget(), terminated_by=RunTermination.COMPLETED,
            wall_clock_s=260.0,
        )

    summary = _last_line(path)
    state_total = 0.006 + 0.007 + 0.00768
    disk_total = 0.006 + 0.007 + 0.00743 + 0.00768
    assert summary["total_usd"] == pytest.approx(disk_total)
    assert summary["total_usd"] > state_total
    assert summary["cost_ledger_gap_usd"] == pytest.approx(0.00743)
    assert summary["n_events"] == 4  # the disk count, not the state count (3)
    assert any("disk-logged cost" in r.message for r in caplog.records)


def test_reconciliation_never_undercounts_relative_to_disk(tmp_path):
    """Sanity check on the max(): even a bizarre state total larger than
    disk (which should never happen in practice) must not cause total_usd
    to fall BELOW what's actually on disk."""
    path = cost_log._COST_LOG_PATH
    path.write_text(_disk_line("run-3", "bull_turn:0:aaa", 0.01) + "\n")
    events = [_event("bull_turn", 0.01, "bull_turn:0:aaa")]

    cost_log.log_run_summary(
        run_id="run-3", ticker="ACN", as_of_date=date(2026, 8, 26),
        events=events, budget=_budget(), terminated_by=RunTermination.COMPLETED,
        wall_clock_s=5.0,
    )

    summary = _last_line(path)
    assert summary["total_usd"] >= 0.01
