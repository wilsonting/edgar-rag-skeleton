"""Phase 9 §3 — run the battery, and write the manifest as it goes.

Sequential and in the foreground, one ticker at a time. Not parallel: it
muddies the per-run cost attribution, risks provider rate limits, and turns
a hang into a question about which of six runs is stuck.

A single failure must NOT abort the battery. Five data points and one
recorded failure is a result; one data point and an aborted script is not.
Each run's RunRecord is appended and the manifest rewritten immediately
after that run, so a battery killed at ticker four still leaves a valid
manifest describing three completed runs and one interrupted one.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date, datetime, timezone
from importlib import metadata
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

import os  # noqa: E402

from app.infrastructure.llm.models import model_env_vars
from app.agent.trading.domain.validation import (  # noqa: E402
    BatteryManifest,
    RunRecord,
)

WATCHLIST = ["NFLX", "AVGO", "ACN", "FIG", "ASML", "MSFT"]
COST_LOG = Path("docs/cost-log.jsonl")

# The env vars that actually select a model. Recorded rather than a
# hand-written "Haiku nodes / Sonnet nodes" tiering, which is a thing the
# code does not know and so cannot be checked against on a rerun.
#
# Sourced from the role registry rather than listed here: the hand-written
# copy had drifted, carrying two variables nothing read (RISK_MODEL,
# RISK_JUDGE_MODEL) while missing the ones that do select a model. A run
# recorded against the wrong list is a run whose configuration cannot be
# reconstructed.
MODEL_ENV_VARS = [*model_env_vars(), "OPENAI_MODEL", "EMBEDDING_MODEL"]

PINNED_PACKAGES = ["langgraph", "pandas-ta-classic", "anthropic", "pydantic"]

_VAULT_RE = re.compile(r"^\[vault\] run .*?: (.+)$", re.M)
_RAW_MEMO_RE = re.compile(r"## Raw memo\s*\n+```json\n(.*?)\n```", re.S)


def preflight() -> list[str]:
    """Service preconditions, checked ONCE before the loop rather than
    discovered six subprocess launches later.

    The one that actually bit (2026-08-27, first battery attempt): the
    research agent behind `fundamentals_node` does not talk to the database
    directly — it calls tools over HTTP against the FastAPI app at
    `app.agent.tools.API_BASE`. With the server down, all six runs died in
    `check_corpus` with `httpx.ConnectError` before spending a cent. Every
    prior battery in this repo used `MOCK_FUNDAMENTALS=1`, which returns
    from a cache file before any tool call, so no earlier phase ever
    needed the server and no earlier gate ever checked for it.

    Cheap, and it fails the battery in a second with a sentence that names
    the fix instead of six stack traces that name httpx.
    """
    problems = []

    from app.agent.tools import API_BASE

    try:
        import httpx

        httpx.get(f"{API_BASE}/corpus-status", params={"ticker": "MSFT"}, timeout=5.0)
    except Exception as exc:  # noqa: BLE001 — any failure to reach it is the same answer
        problems.append(
            f"the research agent's API at {API_BASE} is unreachable ({type(exc).__name__}). "
            f"fundamentals_node calls it for every tool; start it with:\n"
            f"    uv run uvicorn app.main:app --host 127.0.0.1 --port 8000"
        )

    if os.getenv("MOCK_FUNDAMENTALS", "").strip() == "1":
        problems.append(
            "MOCK_FUNDAMENTALS=1 is set. The battery would load cached "
            "fundamentals for the five tickers that have a cache file and run "
            "the real agent only for the one that does not — five cached memos "
            "and one live one is not a cross-ticker comparison. Unset it."
        )

    return problems


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def _package_versions() -> dict[str, str]:
    out = {}
    for name in PINNED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def _run_summary(run_id: str) -> dict | None:
    """The LAST run_summary line for this run_id. Last, not first: a resumed
    run writes a second line, and the later one is the one describing the
    invocation that just finished."""
    if not COST_LOG.exists():
        return None
    found = None
    with COST_LOG.open() as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("kind") == "run_summary" and entry.get("run_id") == run_id:
                found = entry
    return found


def _memo_from_vault(folder: Path, ticker: str) -> tuple[Path | None, dict | None, bool]:
    """Returns (path, memo, verification_failed).

    A `*-decision_failed.md` is what `save_failed_decision_memo` writes when
    `verify_decision_memo` rejected the assembled memo — the run then raises
    and exits non-zero. That artifact IS criterion 4's signal, which is why
    it is recorded here rather than inferred from the exit code alone.
    """
    failed = folder / f"{ticker}-decision_failed.md"
    ok = folder / f"{ticker}-decision.md"
    path = ok if ok.exists() else (failed if failed.exists() else None)
    if path is None:
        return None, None, False
    m = _RAW_MEMO_RE.search(path.read_text())
    memo = json.loads(m.group(1)) if m else None
    return path, memo, path == failed


def run_one(ticker: str, thread_id: str, as_of: date, out_dir: Path,
            max_usd: float, timeout_s: float) -> RunRecord:
    record = RunRecord(
        ticker=ticker,
        thread_id=thread_id,
        as_of_date=as_of,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    cmd = [
        "uv", "run", "python", "-m", "app.agent.trading.interface.cli", ticker,
        "--as-of", as_of.isoformat(),
        "--thread-id", thread_id,
        "--max-usd", str(max_usd),
        "--wall-clock-timeout-s", str(timeout_s),
    ]
    print(f"\n=== {ticker}  ({thread_id}) ===", flush=True)
    stdout_path = out_dir / f"{ticker}.stdout"
    stderr_path = out_dir / f"{ticker}.stderr"
    with stdout_path.open("w") as so, stderr_path.open("w") as se:
        proc = subprocess.run(cmd, stdout=so, stderr=se, text=True)
    record.exit_code = proc.returncode
    record.finished_at = datetime.now(timezone.utc).isoformat()

    stdout = stdout_path.read_text()
    summary = _run_summary(thread_id)
    if summary:
        record.total_usd = summary.get("total_usd")
        record.cost_ledger_gap_usd = summary.get("cost_ledger_gap_usd")
        record.cache_read_ratio = summary.get("cache_read_ratio")
        record.wall_clock_s = summary.get("wall_clock_s")
        terminated = summary.get("terminated_by")
        if terminated in ("budget_exceeded", "deadline_exceeded"):
            record.exit_status = terminated
        elif proc.returncode == 0:
            record.exit_status = "ok"
    elif proc.returncode == 0:
        record.exit_status = "ok"

    if proc.returncode != 0:
        tail = stderr_path.read_text().strip().splitlines()
        record.error_repr = tail[-1] if tail else f"exit {proc.returncode}"
        if record.exit_status == "ok":
            record.exit_status = "error"

    vault = _VAULT_RE.search(stdout)
    if vault:
        folder = Path(vault.group(1))
        record.run_folder = str(folder)
        path, memo, failed = _memo_from_vault(folder, ticker.upper())
        record.memo_verification_failed = failed
        if path is not None:
            record.memo_md_path = str(path)
        if memo is not None:
            record.verdict = memo.get("verdict")
            record.verdict_samples = memo.get("verdict_samples") or []
            record.confidence = memo.get("confidence")
            record.data_as_of_date = memo.get("data_as_of_date")
            record.n_data_gaps = len(memo.get("data_gaps") or [])

    print(
        f"--- {ticker}: exit={record.exit_code} status={record.exit_status} "
        f"verdict={record.verdict} conf={record.confidence} "
        f"usd={record.total_usd} wall={record.wall_clock_s}s",
        flush=True,
    )
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", type=date.fromisoformat, required=True)
    ap.add_argument("--tickers", nargs="*", default=WATCHLIST)
    ap.add_argument("--suffix", default="", help="thread-id suffix, e.g. -r2 for the §7 re-runs")
    ap.add_argument("--max-usd", type=float, default=0.75)
    ap.add_argument("--wall-clock-timeout-s", type=float, default=1800.0)
    args = ap.parse_args()

    battery_id = f"p9-{args.as_of.isoformat().replace('-', '')}"
    out_dir = Path("docs/validation") / battery_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"manifest{args.suffix or ''}.json"

    dirty = bool(_git("status", "--porcelain"))
    if dirty:
        print("WARNING: working tree is dirty — this battery is not reproducible "
              "from the recorded SHA alone.", file=sys.stderr)

    manifest = BatteryManifest(
        battery_id=battery_id,
        as_of_date=args.as_of,
        git_sha=_git("rev-parse", "HEAD"),
        git_dirty=dirty,
        model_ids={v: os.environ.get(v, "<unset>") for v in MODEL_ENV_VARS},
        package_versions=_package_versions(),
        max_usd=args.max_usd,
        wall_clock_timeout_s=args.wall_clock_timeout_s,
    )

    # Carry forward rows for tickers this invocation is NOT running.
    #
    # Without this, re-running one ticker rebuilt the manifest from scratch
    # and destroyed the rest of the battery's record. Live (2026-08-28): a
    # `--tickers MSFT` invocation replaced a four-ticker manifest with a
    # one-ticker one. The evidence itself survived — vault memos, cost log,
    # checkpoints are all elsewhere — but the manifest is what §4's gate and
    # §7's comparison read, so the battery looked like it had one run.
    #
    # A ticker being re-run is REPLACED, not appended: the newest attempt is
    # the one that describes the current state of that thread, and keeping
    # both would make "how many runs completed" ambiguous.
    if manifest_path.exists():
        try:
            prior = BatteryManifest.model_validate_json(manifest_path.read_text())
        except Exception as exc:  # noqa: BLE001 — a corrupt manifest must not block a run
            print(f"WARNING: could not read {manifest_path} ({exc}); starting fresh",
                  file=sys.stderr)
        else:
            rerunning = set(args.tickers)
            carried = [r for r in prior.runs if r.ticker not in rerunning]
            manifest.runs.extend(carried)
            if carried:
                print(f"carrying forward {len(carried)} prior run(s): "
                      f"{', '.join(r.ticker for r in carried)}")

    problems = preflight()
    if problems:
        print("PREFLIGHT FAILED — nothing was run:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    for ticker in args.tickers:
        thread_id = f"trading-{ticker}-{battery_id}{args.suffix}"
        manifest.runs.append(
            run_one(ticker, thread_id, args.as_of, out_dir,
                    args.max_usd, args.wall_clock_timeout_s)
        )
        # Rewritten after EVERY run, not once at the end: a battery killed
        # at ticker four should still leave a manifest describing three.
        manifest_path.write_text(manifest.model_dump_json(indent=2))

    total = sum(r.total_usd or 0.0 for r in manifest.runs)
    ok = sum(1 for r in manifest.runs if r.exit_status == "ok")
    print(f"\n{ok}/{len(manifest.runs)} runs ok · ${total:.4f} total · {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
