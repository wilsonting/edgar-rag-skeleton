"""Reconstruct battery manifest rows from the artifacts that outlive it.

The manifest is a convenience index, not the system of record: the vault
memos, `docs/cost-log.jsonl` and the LangGraph checkpoints all persist
independently. So a lost or clobbered manifest is recoverable, and this
recovers it — used after a `--tickers X` invocation rebuilt the file from
scratch and dropped four completed runs (2026-08-28; the runner now carries
prior rows forward, see run_p9_battery.py).

Reconstructed rows carry `started_at` from the cost log's run_summary
timestamp rather than the real start instant, which is not recorded
anywhere. Everything else is exact.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

from app.agent.trading.domain.validation import BatteryManifest, RunRecord

COST_LOG = Path("docs/cost-log.jsonl")
_RAW = re.compile(r"## Raw memo\s*\n+```json\n(.*?)\n```", re.S)


def _summaries() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with COST_LOG.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("kind") == "run_summary":
                out[e["run_id"]] = e  # last wins: the newest attempt
    return out


def _memo_for(vault: Path, ticker: str, thread_id: str) -> tuple[Path, dict] | None:
    """Newest decision memo for this ticker. Matched by recency rather than
    by thread id, which the vault path does not record — safe here because a
    battery writes one run per ticker per day."""
    candidates = sorted(vault.glob(f"{ticker}/*/*/{ticker}-decision.md"))
    if not candidates:
        return None
    path = candidates[-1]
    m = _RAW.search(path.read_text())
    return (path, json.loads(m.group(1))) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--vault", type=Path,
                    default=Path.home() / "Documents/Obsidian Vault/EDGAR-MEMO/memos")
    args = ap.parse_args()

    manifest = BatteryManifest.model_validate_json(args.manifest.read_text())
    summaries = _summaries()
    have = {r.ticker for r in manifest.runs}

    for ticker in args.tickers:
        if ticker in have:
            print(f"{ticker}: already present, skipping")
            continue
        thread_id = f"trading-{ticker}-{manifest.battery_id}-a2"
        s = summaries.get(thread_id)
        if s is None:
            print(f"{ticker}: no run_summary for {thread_id}, skipping")
            continue
        found = _memo_for(args.vault, ticker, thread_id)
        record = RunRecord(
            ticker=ticker,
            thread_id=thread_id,
            as_of_date=manifest.as_of_date,
            started_at=s["timestamp"],
            finished_at=s["timestamp"],
            exit_code=0 if s["terminated_by"] == "completed" else 1,
            exit_status="ok" if s["terminated_by"] == "completed" else s["terminated_by"],
            total_usd=s["total_usd"],
            cost_ledger_gap_usd=s.get("cost_ledger_gap_usd"),
            cache_read_ratio=s.get("cache_read_ratio"),
            wall_clock_s=s["wall_clock_s"],
        )
        if found:
            path, memo = found
            record.memo_md_path = str(path)
            record.run_folder = str(path.parent)
            record.verdict = memo.get("verdict")
            record.verdict_samples = memo.get("verdict_samples") or []
            record.confidence = memo.get("confidence")
            # Coerced explicitly: pydantic does not validate on attribute
            # assignment, so a raw string would sit in a `date` field and
            # serialize with a warning.
            raw_as_of = memo.get("data_as_of_date")
            record.data_as_of_date = date.fromisoformat(raw_as_of) if raw_as_of else None
            record.n_data_gaps = len(memo.get("data_gaps") or [])
        manifest.runs.append(record)
        print(f"{ticker}: rebuilt — {record.exit_status}, {record.verdict}, "
              f"${record.total_usd:.4f}")

    manifest.runs.sort(key=lambda r: r.ticker)
    args.manifest.write_text(manifest.model_dump_json(indent=2))
    print(f"\n{len(manifest.runs)} run(s) in {args.manifest}")


if __name__ == "__main__":
    main()
