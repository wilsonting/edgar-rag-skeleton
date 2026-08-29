"""Turns an already-logged LLM call into the `CostEvent` that rides in
`TradingState.cost_events`, and writes the run-level summary line.

Deliberately NOT a wrapper around `researcher.log_cost` — every port module
already calls `log_cost(...)` directly, and several tests monkeypatch that
exact call (`monkeypatch.setattr(port, "log_cost", ...)`) to keep from
writing to the real `docs/cost-log.jsonl` during a run that never touches
the network. A wrapper that called `log_cost` from a second module would
bypass those patches — the module a name is imported into, not the module it
was defined in, is what monkeypatch replaces. So `record_cost_event` takes
the cost `log_cost` already computed as a plain argument instead of
recomputing or re-logging it: one number, two representations (the disk line
and the state event), never two sources of truth for it.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime
from pathlib import Path

from app.agent.researcher import UsageSummary
from app.agent.trading.domain.budget import CostEvent, RunBudget, RunTermination, total_spend

logger = logging.getLogger(__name__)

_COST_LOG_PATH = Path("docs/cost-log.jsonl")

# The growing-transcript stages criterion 3's cache-read ratio is about —
# NOT the analyst or synthesis nodes, whose evidence pack is sent once, not
# re-sent turn over turn.
_DEBATE_STAGE_NODES = frozenset(
    {"bull_turn", "bear_turn", "neutral_turn", "aggressive_turn", "conservative_turn"}
)


def new_event_id(node: str, *, turn_index: int | None = None) -> str:
    """Generated BEFORE the sibling `log_cost(...)` call at each site, and
    passed to both it and `record_cost_event` below — so the disk line and
    the state event carry the identical id. (First measured live run showed
    why this has to be a separate, earlier step: generating the id inside
    `record_cost_event`, called AFTER `log_cost`, left every disk line's
    `event_id` null — the two were never the same string.)"""
    suffix = f":{turn_index}" if turn_index is not None else ""
    return f"{node}{suffix}:{uuid.uuid4().hex[:8]}"


def record_cost_event(
    event_id: str,
    node: str,
    usage: UsageSummary,
    model: str,
    cost: float | None,
) -> CostEvent:
    """`cost` is whatever the sibling `log_cost(...)` call at the same call
    site already returned — never recomputed here, so the state event and
    the disk line can never disagree about the dollar amount. `event_id`
    comes from `new_event_id()`, called once per site and passed to both."""
    return CostEvent(
        event_id=event_id,
        node=node,
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_input_tokens=usage.cache_write_tokens,
        cache_read_input_tokens=usage.cache_read_tokens,
        usd=cost if cost is not None else 0.0,
    )


def _disk_logged_events(run_id: str) -> list[dict]:
    """Every `cost_event` line already on disk for this run_id.

    Exists because `TradingState.cost_events` can UNDER-count real spend —
    live-verified (Phase 8 criterion 7, crash-resume test): `log_cost`
    writes its disk line synchronously, before the calling node returns,
    but a node that then crashes (or is retried by LangGraph after a
    transient failure) never returns its delta, so that already-billed
    call's CostEvent never reaches `cost_events`. The disk log is a strict
    superset of state in that scenario, never a duplicate of it — the
    retried call gets its own fresh `event_id` and bills again for real —
    so summing disk lines is the conservative, honest total, not a
    double-count risk the way trusting `cost_events` alone is an
    under-count risk.
    """
    if not _COST_LOG_PATH.exists():
        return []
    events: list[dict] = []
    with _COST_LOG_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("kind") == "cost_event" and entry.get("run_id") == run_id:
                events.append(entry)
    return events


def _cache_read_ratio(events: list[CostEvent]) -> float | None:
    debate_stage = [e for e in events if e.node in _DEBATE_STAGE_NODES]
    if not debate_stage:
        return None
    read = sum(e.cache_read_input_tokens for e in debate_stage)
    denom = read + sum(
        e.input_tokens + e.cache_creation_input_tokens for e in debate_stage
    )
    return round(read / denom, 4) if denom else None


def log_run_summary(
    *,
    run_id: str,
    ticker: str,
    as_of_date: date,
    events: list[CostEvent],
    budget: RunBudget,
    terminated_by: RunTermination,
    wall_clock_s: float,
) -> None:
    """Written exactly once per run — completed or aborted — right after the
    vault artifacts save in cli.py. That one-line-per-run invariant is what
    makes criterion 1 ("every run_summary shows total_usd <= 0.60") a single
    `jq` query rather than a reconstruction from per-call lines.

    `total_usd` is reconciled against the disk log itself, not taken from
    `state["cost_events"]` alone (Phase 8 criterion 7 finding): a node that
    crashes or is retried after its LLM call already committed a cost-log
    line can leave that call's CostEvent out of `cost_events` forever, even
    though it was genuinely billed. The disk log never loses a call the
    state ledger has (log_cost writes before the node can fail), so taking
    the larger of the two is always at least as accurate, never a
    double-count — a retried call bills again for real and gets its own
    `event_id`, it doesn't duplicate the crashed one. `cost_ledger_gap_usd`
    makes a reconciliation visible rather than silently correcting the
    number — same flag-not-assert posture as `data_gaps`/`guard_flags`
    elsewhere in this pipeline.
    """
    state_total = round(total_spend(events), 6)
    disk_events = _disk_logged_events(run_id)
    disk_total = round(sum(e.get("estimated_cost_usd") or 0.0 for e in disk_events), 6)

    gap = round(disk_total - state_total, 6)
    if gap > 0:
        logger.warning(
            "run %s: disk-logged cost $%.6f exceeds the state-derived total "
            "$%.6f by $%.6f — a node was retried or crashed after an LLM "
            "call committed cost to disk but before TradingState.cost_events "
            "saw it. Reporting the disk total.",
            run_id, disk_total, state_total, gap,
        )
    total_usd = max(state_total, disk_total)
    n_events = len(disk_events) if gap > 0 else len(events)

    entry = {
        "kind": "run_summary",
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "ticker": ticker,
        "as_of_date": as_of_date.isoformat(),
        "total_usd": total_usd,
        "cost_ledger_gap_usd": max(gap, 0.0),
        "budget_max_usd": budget.max_usd,
        "terminated_by": terminated_by.value,
        "cache_read_ratio": _cache_read_ratio(events),
        "n_events": n_events,
        "wall_clock_s": round(wall_clock_s, 3),
    }
    _COST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _COST_LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
