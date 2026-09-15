"""Where the cost log lives, and how it is rotated.

`Path("docs/cost-log.jsonl")` was written literally in two modules —
`researcher.log_cost` and `trading/infrastructure/cost_log` — with no shared
constant. Both were relative to the process working directory, so a run
started anywhere but the repo root silently created a new `docs/` there and
wrote a log nothing ever reconciled against. The reconciliation in
`log_run_summary` reads the file back to catch spend that a crashed node
never returned to state; pointed at a different file, it reads nothing and
reports a clean run.

Resolved from THIS file's location instead, so the log follows the
repository rather than the shell. `COST_LOG_DIR` overrides it.

Rotation is by month. The file is append-only and had reached 3,177 lines /
1.0 MB, and `_disk_logged_events` parses the whole thing on every run
summary. Monthly files keep that bounded without losing history: the reader
takes the current month and the previous one, which covers a run that starts
on the last day of a month and ends on the first of the next.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# The pre-rotation file. Runs before this change wrote every line here, and
# the reconciliation still reads it so their history stays reachable.
LEGACY_LOG_NAME = "cost-log.jsonl"


def cost_log_dir() -> Path:
    return Path(os.getenv("COST_LOG_DIR") or _REPO_ROOT / "docs")


def cost_log_path(on: date | None = None) -> Path:
    """The file a line written now belongs in."""
    day = on or date.today()
    return cost_log_dir() / f"cost-log-{day:%Y-%m}.jsonl"


def readable_logs(on: date | None = None) -> list[Path]:
    """Every file a run's lines could be in, newest last.

    The current month, the previous one (a run can straddle midnight on the
    1st), and the pre-rotation file.
    """
    day = on or date.today()
    previous = (day.replace(day=1) - __import__("datetime").timedelta(days=1))
    candidates = [
        cost_log_dir() / LEGACY_LOG_NAME,
        cost_log_path(previous),
        cost_log_path(day),
    ]
    seen: list[Path] = []
    for path in candidates:
        if path not in seen and path.exists():
            seen.append(path)
    return seen
