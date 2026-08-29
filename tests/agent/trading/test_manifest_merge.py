"""Re-running one ticker must not destroy the rest of the battery's record.

Live (2026-08-28): a `--tickers MSFT` invocation rebuilt the manifest from
scratch and replaced four completed runs with one. The evidence survived —
vault memos, cost log and checkpoints are all elsewhere — but the manifest
is what the automated gate and the stability comparison read, so the battery
looked like it had a single run.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from app.agent.trading.domain.validation import BatteryManifest, RunRecord


def _manifest(*tickers: str) -> BatteryManifest:
    m = BatteryManifest(
        battery_id="p9-20260826", as_of_date=date(2026, 8, 26), git_sha="abc",
        git_dirty=False, model_ids={}, package_versions={},
        max_usd=1.40, wall_clock_timeout_s=1800.0,
    )
    for t in tickers:
        m.runs.append(RunRecord(
            ticker=t, thread_id=f"trading-{t}-p9", as_of_date=date(2026, 8, 26),
            started_at="x", exit_status="ok", verdict="hold", total_usd=0.8,
        ))
    return m


def _carry_forward(prior: BatteryManifest, fresh: BatteryManifest, rerunning: set[str]):
    """Mirrors the runner's merge rule, kept here so the RULE is tested even
    though the runner reads its prior state from disk."""
    fresh.runs.extend(r for r in prior.runs if r.ticker not in rerunning)
    return fresh


def test_untouched_tickers_are_carried_forward():
    prior = _manifest("NFLX", "AVGO", "ACN", "FIG")
    fresh = _manifest()

    merged = _carry_forward(prior, fresh, {"MSFT"})

    assert {r.ticker for r in merged.runs} == {"NFLX", "AVGO", "ACN", "FIG"}


def test_a_rerun_ticker_is_replaced_not_duplicated():
    """Keeping both attempts would make "how many runs completed" ambiguous;
    the newest attempt is the one describing the thread's current state."""
    prior = _manifest("NFLX", "MSFT")
    fresh = _manifest("MSFT")

    merged = _carry_forward(prior, fresh, {"MSFT"})

    assert [r.ticker for r in merged.runs].count("MSFT") == 1
    assert {r.ticker for r in merged.runs} == {"NFLX", "MSFT"}


def test_a_full_battery_rerun_carries_nothing_forward():
    prior = _manifest("NFLX", "AVGO")
    fresh = _manifest("NFLX", "AVGO")

    merged = _carry_forward(prior, fresh, {"NFLX", "AVGO"})

    assert len(merged.runs) == 2


def test_the_merged_manifest_still_round_trips():
    merged = _carry_forward(_manifest("NFLX"), _manifest("MSFT"), {"MSFT"})
    assert BatteryManifest.model_validate_json(merged.model_dump_json()) == merged


def test_the_real_manifest_on_disk_is_intact(tmp_path):
    """Guards the reconstruction itself: the battery's manifest should hold
    every ticker that was run, not just the last one."""
    from pathlib import Path
    p = Path("docs/validation/p9-20260826/manifest-a2.json")
    if not p.exists():
        pytest.skip("battery manifest not present in this checkout")
    m = BatteryManifest.model_validate_json(p.read_text())
    assert {r.ticker for r in m.runs} >= {"NFLX", "AVGO", "ACN", "FIG"}
