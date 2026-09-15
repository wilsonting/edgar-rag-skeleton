"""What code is actually running, so a stale process can say so.

A 22-hour-old `uvicorn` served a full FIG pipeline run on 2026-09-13. It
predated every fix the run was meant to verify, and nothing anywhere said
so: the CLI ran current code, the server ran September 12th's, and the only
symptom was a date bound that silently did nothing. A whole run's worth of
verification was spent before the mismatch was noticed.

Read once at import — that is the point. The value is the commit the
RUNNING PROCESS started with, not whatever the working tree says now, so it
must not be recomputed per request.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO), *args],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        # No git, no repo, no binary — a deployment detail, not a failure.
        return None


def _head() -> str | None:
    return _git("rev-parse", "--short", "HEAD")


def _dirty() -> bool:
    return bool(_git("status", "--porcelain"))


# Snapshotted at import: this is what the process is running.
COMMIT: str | None = _head()
DIRTY: bool = _dirty()


def build_info() -> dict:
    return {"commit": COMMIT, "dirty": DIRTY}


def describe() -> str:
    if COMMIT is None:
        return "unknown"
    return f"{COMMIT}{'-dirty' if DIRTY else ''}"


def working_tree_matches() -> tuple[bool, str]:
    """Whether the working tree is still what this process started with.

    Returns (matches, message). A client calls this against a server's
    reported commit; the server calls it against its own to notice that it
    has been left running across an edit.
    """
    current = _head()
    if COMMIT is None or current is None:
        return True, "no git metadata; cannot compare"
    if current == COMMIT:
        return True, f"running {describe()}"
    return False, (
        f"this process started at {COMMIT} but the working tree is now at "
        f"{current} — restart it to pick up the change"
    )
