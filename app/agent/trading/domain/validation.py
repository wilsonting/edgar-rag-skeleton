"""The Phase 9 validation battery's own record of what it ran.

A battery is an experiment, and an experiment that does not record its
inputs is not reproducible — a rerun after a fix is then a different
experiment wearing the same name. `BatteryManifest` is that record: the
tree, the models, the one `as_of_date`, and one `RunRecord` per invocation.

Field names track the pipeline's real vocabulary rather than a generic
one: the memo's decision is `verdict` (a `Verdict`, four-valued — see
`decision_memo.py` for why `unresolved` exists), the memo's date field is
`data_as_of_date`, and the run's stop reason is a `RunTermination`. A
manifest that renamed these would need a translation layer between it and
every artifact it describes, and the translation is where drift starts.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# `error` is this module's own addition and has no `RunTermination` member:
# the three RunTermination values all describe a run the graph itself ended
# deliberately, whereas a crash, a MemoVerificationError, or a non-zero exit
# for any other reason is the battery runner observing a run that stopped
# without the graph choosing to. Keeping them in one field, distinguishable,
# is what lets criterion 2 ("six invocations exit 0") be read off the
# manifest instead of reconstructed from stderr.
ExitStatus = Literal["ok", "error", "budget_exceeded", "deadline_exceeded"]


class RunRecord(BaseModel):
    ticker: str
    thread_id: str
    as_of_date: date
    started_at: str  # ISO-8601
    finished_at: str | None = None
    exit_code: int | None = None
    exit_status: ExitStatus = "error"
    error_repr: str | None = None

    # Vault paths. The pipeline writes one folder per run (see
    # researcher._save_output), so these are recorded rather than derived —
    # a folder name carries the run's start instant, which the manifest
    # writer cannot recompute afterwards without guessing.
    run_folder: str | None = None
    memo_md_path: str | None = None
    # True when the run produced a `*-decision_failed.md` instead of a
    # `*-decision.md`. That artifact only exists when `verify_decision_memo`
    # rejected the assembled memo, which is criterion 4's actual signal —
    # see `p9_automated_gate.py` for why the criterion cannot be re-checked
    # standalone in this architecture.
    memo_verification_failed: bool = False

    total_usd: float | None = None
    cost_ledger_gap_usd: float | None = None
    cache_read_ratio: float | None = None
    wall_clock_s: float | None = None

    # Read off the memo, not off stdout — §7's stability comparison joins on
    # these, and re-parsing six memos later to recover them is work the
    # battery already did once.
    verdict: str | None = None
    verdict_samples: list[str] = Field(default_factory=list)
    confidence: float | None = None
    data_as_of_date: date | None = None
    n_data_gaps: int | None = None


class BatteryManifest(BaseModel):
    # `model_ids` collides with pydantic's protected `model_` namespace. The
    # field is named for the thing it holds, not around a framework rule.
    model_config = ConfigDict(protected_namespaces=())

    battery_id: str  # e.g. "p9-20260825"
    as_of_date: date
    git_sha: str
    git_dirty: bool
    # Every model the run can reach, by the env var that selects it — the
    # tier names ("Haiku nodes") are not a thing the code knows, but the
    # variables are, and they are what a rerun has to match.
    model_ids: dict[str, str]
    package_versions: dict[str, str]
    max_usd: float
    wall_clock_timeout_s: float
    runs: list[RunRecord] = Field(default_factory=list)

    def by_ticker(self, ticker: str) -> list[RunRecord]:
        return [r for r in self.runs if r.ticker == ticker]


# --- §7 stability: direction, not exact enum -------------------------------
#
# `Verdict` has four values and a stochastic pipeline is not expected to
# reproduce the exact one. Direction is the property a reader would act on,
# so that is what criterion 8 compares. `unresolved` maps to "neither"
# alongside `hold`: both mean the battery is not telling you to move, and
# collapsing them keeps a hold->unresolved drift from reading as a flip when
# nothing actionable changed. (It does mean a genuine hold->unresolved
# change is invisible here — `verdict_samples` in the manifest is where that
# shows up, and it is why the raw samples are recorded.)
_DIRECTION = {
    "buy": "bullish",
    "sell": "bearish",
    "hold": "neither",
    "unresolved": "neither",
}


def verdict_direction(verdict: str | None) -> str:
    if verdict is None:
        return "unknown"
    return _DIRECTION.get(verdict.lower(), "unknown")
