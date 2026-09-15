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
    """The log directory is now resolved from the repo (and overridable),
    rather than being a module constant tests could patch."""
    monkeypatch.setenv("COST_LOG_DIR", str(tmp_path))
    return _log_path()


def _log_path():
    from app.infrastructure.cost_log_path import cost_log_path

    return cost_log_path()


def _last_line(path) -> dict:
    return json.loads(path.read_text().strip().splitlines()[-1])


def test_no_gap_when_disk_matches_state(tmp_path):
    path = _log_path()
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
    path = _log_path()
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
    path = _log_path()
    path.write_text(_disk_line("run-3", "bull_turn:0:aaa", 0.01) + "\n")
    events = [_event("bull_turn", 0.01, "bull_turn:0:aaa")]

    cost_log.log_run_summary(
        run_id="run-3", ticker="ACN", as_of_date=date(2026, 8, 26),
        events=events, budget=_budget(), terminated_by=RunTermination.COMPLETED,
        wall_clock_s=5.0,
    )

    summary = _last_line(path)
    assert summary["total_usd"] >= 0.01


@pytest.mark.parametrize("resumed", [False, True])
def test_the_summary_says_whether_it_covers_a_resumed_run(resumed):
    cost_log.log_run_summary(
        run_id="run-r", ticker="ACN", as_of_date=date(2026, 8, 26),
        events=[], budget=_budget(), terminated_by=RunTermination.COMPLETED,
        wall_clock_s=1.0, resumed=resumed,
    )
    assert _last_line(_log_path())["resumed"] is resumed


# ---------------------------------------------------------------------------
# Where the log lives
# ---------------------------------------------------------------------------

def test_the_path_follows_the_repo_not_the_working_directory(monkeypatch, tmp_path):
    """It was Path("docs/cost-log.jsonl") in two modules. A run started
    anywhere but the repo root wrote a log the disk reconciliation in
    log_run_summary never read — and that reconciliation is what catches
    spend a crashed node never returned to state."""
    import os
    from app.infrastructure.cost_log_path import cost_log_dir

    monkeypatch.delenv("COST_LOG_DIR", raising=False)
    here = cost_log_dir()
    monkeypatch.chdir(tmp_path)
    assert cost_log_dir() == here
    assert here.is_absolute()


def test_one_file_per_month(monkeypatch):
    from datetime import date

    from app.infrastructure.cost_log_path import cost_log_path

    monkeypatch.delenv("COST_LOG_DIR", raising=False)
    assert cost_log_path(date(2026, 9, 12)).name == "cost-log-2026-09.jsonl"
    assert cost_log_path(date(2026, 1, 31)).name == "cost-log-2026-01.jsonl"


def test_a_run_that_straddles_a_month_boundary_is_still_reconcilable(monkeypatch, tmp_path):
    """log_run_summary reads the log back to find spend state lost. A run
    that starts on the 31st and ends on the 1st has lines in two files."""
    from datetime import date

    from app.infrastructure.cost_log_path import readable_logs

    monkeypatch.setenv("COST_LOG_DIR", str(tmp_path))
    (tmp_path / "cost-log-2026-08.jsonl").write_text("")
    (tmp_path / "cost-log-2026-09.jsonl").write_text("")
    names = [p.name for p in readable_logs(date(2026, 9, 1))]
    assert names == ["cost-log-2026-08.jsonl", "cost-log-2026-09.jsonl"]


def test_the_pre_rotation_file_stays_reachable(monkeypatch, tmp_path):
    """Every run before this change wrote to docs/cost-log.jsonl."""
    from datetime import date

    from app.infrastructure.cost_log_path import readable_logs

    monkeypatch.setenv("COST_LOG_DIR", str(tmp_path))
    (tmp_path / "cost-log.jsonl").write_text("")
    assert [p.name for p in readable_logs(date(2026, 9, 12))] == ["cost-log.jsonl"]


def test_reconciliation_finds_a_run_split_across_rotated_files(monkeypatch, tmp_path):
    """A run whose lines landed in the pre-rotation file and this month's."""
    from app.agent.trading.infrastructure.cost_log import _disk_logged_events

    monkeypatch.setenv("COST_LOG_DIR", str(tmp_path))
    (tmp_path / "cost-log.jsonl").write_text(_disk_line("r1", "bull_turn:0:a", 0.01) + "\n")
    _log_path().write_text(
        _disk_line("r1", "bear_turn:1:b", 0.02) + "\n"
        + _disk_line("r2", "bull_turn:0:c", 9.0) + "\n"
    )

    found = _disk_logged_events("r1")
    assert sum(e["estimated_cost_usd"] for e in found) == pytest.approx(0.03)
