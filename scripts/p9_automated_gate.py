"""Phase 9 §4 — every machine-checkable criterion, run before a human reads
a single memo. It is fast, and it says whether the manual audit is even
worth starting.

Criteria 5 and 6 (the manual audit) are the actual deliverable. Everything
here is the scaffolding that makes that audit mean something.

Two criteria differ from the guide's shape because the pipeline differs, and
both differences are recorded rather than papered over:

Criterion 4 ("re-run verify_decision_memo over the six memos as a batch")
cannot be done in this architecture, and re-running it would be WORSE than
not. `verify_decision_memo(memo, state, ledger, debate_turns)` must be given
the SAME trial the memo came from — under majority-of-N sampling each trial
runs its own risk panel with its own content-hashed factor ids, and the
winning memo is frequently not from the trial whose panel is left in the
checkpoint. Verifying against a different trial's ledger would report real
citations as unresolved: a false failure, not a check. What the pipeline
does instead is stronger: `synthesizer_node` calls the verifier on the final
memo against its own trial and RAISES on failure, saving a
`*-decision_failed.md`. So a `*-decision.md` on disk is proof of a pass, and
criterion 4 is checked by the absence of the failed artifact plus a non-zero
memo count — in-band, at generation time, against the right trial.

Criterion 7 ("no evidence item has a source date after as_of_date") has no
per-item date to check: `DecisionMemo.evidence` is `list[str]`, rendered by
`_render_evidence` from resolved claim/factor references, and neither
`DebateClaim` nor `RiskLedgerEntry` carries a source date. The dated
boundary that DOES exist is enforced structurally upstream — `as_of_date` is
set once at the CLI boundary and the news window is bounded at or before it.
So what is checked here is what is checkable: every memo carries the SAME
`data_as_of_date`, that date equals the battery's, and no four-digit year
later than it appears in the memo's prose. The last is a coarse net for a
lookahead smuggled in as text, and it is reported as an audit lead, not as a
criterion verdict.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from app.agent.trading.domain.decision_memo import DecisionMemo
from app.agent.trading.domain.validation import BatteryManifest, verdict_direction

COST_LOG = Path("docs/cost-log.jsonl")
_RAW_MEMO_RE = re.compile(r"## Raw memo\s*\n+```json\n(.*?)\n```", re.S)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")

BATTERY_CAP_USD = 4.00


class Gate:
    def __init__(self) -> None:
        self.results: list[tuple[int, str, bool, str]] = []

    def record(self, n: int, name: str, passed: bool, detail: str) -> None:
        self.results.append((n, name, passed, detail))

    def report(self) -> int:
        print(f"\n{'#':>3}  {'criterion':38} {'':6} detail")
        print("-" * 100)
        for n, name, passed, detail in self.results:
            print(f"{n:>3}  {name:38} {'PASS' if passed else 'FAIL':6} {detail}")
        failed = [r for r in self.results if not r[2]]
        print()
        if failed:
            print(f"{len(failed)} criterion/criteria FAILED — see §8 triage before auditing.")
            return 1
        print("Automated gate clean. The manual audit (criteria 5-6) is worth starting.")
        return 0


def load_memos(manifest: BatteryManifest) -> dict[str, dict]:
    memos = {}
    for run in manifest.runs:
        if not run.memo_md_path:
            continue
        text = Path(run.memo_md_path).read_text()
        m = _RAW_MEMO_RE.search(text)
        if m:
            memos[run.ticker] = json.loads(m.group(1))
    return memos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    args = ap.parse_args()

    manifest = BatteryManifest.model_validate_json(args.manifest.read_text())
    gate = Gate()
    expected = len(manifest.runs)
    memos = load_memos(manifest)

    # --- 2: every run completed and left a memo --------------------------
    ok = [r for r in manifest.runs if r.exit_status == "ok"]
    with_memo = [r for r in manifest.runs if r.memo_md_path and not r.memo_verification_failed]
    gate.record(
        2, "runs complete + memo on disk",
        len(ok) == expected and len(with_memo) == expected,
        f"{len(ok)}/{expected} exited ok; {len(with_memo)}/{expected} memos"
        + ("" if len(ok) == expected
           else " — failed: " + ", ".join(
               f"{r.ticker}({r.exit_status}: {r.error_repr})" for r in manifest.runs
               if r.exit_status != "ok")),
    )

    # --- 3: schema-valid --------------------------------------------------
    bad = []
    for ticker, memo in memos.items():
        try:
            DecisionMemo.model_validate(memo)
        except Exception as exc:  # noqa: BLE001 — the message is the report
            bad.append(f"{ticker}: {type(exc).__name__}")
    gate.record(3, "DecisionMemo.model_validate", not bad and len(memos) == expected,
                f"{len(memos) - len(bad)}/{expected} valid"
                + (" — " + "; ".join(bad) if bad else ""))

    # --- 4: in-band memo verification (see module docstring) -------------
    failed_verif = [r.ticker for r in manifest.runs if r.memo_verification_failed]
    gate.record(4, "verify_decision_memo (in-band)", not failed_verif and len(memos) == expected,
                f"0 decision_failed artifacts across {expected} runs"
                if not failed_verif
                else "verification failed: " + ", ".join(failed_verif))

    # --- 7: as_of_date integrity -----------------------------------------
    dates = {m.get("data_as_of_date") for m in memos.values()}
    battery_date = manifest.as_of_date.isoformat()
    single = len(dates) == 1 and dates == {battery_date}
    leads = []
    for ticker, memo in memos.items():
        prose = " ".join(
            str(memo.get(f) or "") if not isinstance(memo.get(f), list)
            else " ".join(memo.get(f) or [])
            for f in ("reasoning", "bull_case", "bear_case", "research_thesis",
                      "risk_debate_summary", "technical_signal", "watch_items")
        )
        ahead = sorted({y for y in _YEAR_RE.findall(prose) if y > battery_date[:4]})
        if ahead:
            leads.append(f"{ticker}: mentions {', '.join(ahead)}")
    detail = f"all memos data_as_of_date={battery_date}" if single else f"dates seen: {sorted(dates)}"
    if leads:
        detail += "  |  AUDIT LEADS (not a verdict): " + "; ".join(leads)
    gate.record(7, "single as_of_date, no lookahead", single, detail)

    # --- 9: cost ----------------------------------------------------------
    run_ids = {r.thread_id for r in manifest.runs}
    total = 0.0
    over_cap = []
    for run in manifest.runs:
        total += run.total_usd or 0.0
        if run.total_usd and run.total_usd > manifest.max_usd:
            over_cap.append(f"{run.ticker} ${run.total_usd:.4f}")
    gate.record(9, f"cost <= ${BATTERY_CAP_USD:.2f} battery / ${manifest.max_usd:.2f} run",
                total <= BATTERY_CAP_USD and not over_cap,
                f"${total:.4f} across {len(run_ids)} run(s)"
                + ("" if not over_cap else " — over per-run cap: " + ", ".join(over_cap)))

    # --- context for the audit, not a criterion --------------------------
    print("\nper-run (for §6 audit ordering and §7 ticker selection):")
    print(f"{'ticker':7}{'verdict':11}{'direction':11}{'conf':>6}{'gaps':>6}{'usd':>8}"
          f"{'cache':>7}{'wall_s':>8}  samples")
    for run in sorted(manifest.runs, key=lambda r: (r.confidence is None, r.confidence)):
        print(
            f"{run.ticker:7}{str(run.verdict):11}{verdict_direction(run.verdict):11}"
            f"{run.confidence if run.confidence is not None else float('nan'):>6.2f}"
            f"{run.n_data_gaps if run.n_data_gaps is not None else -1:>6}"
            f"{run.total_usd or 0.0:>8.4f}{run.cache_read_ratio or 0.0:>7.2f}"
            f"{run.wall_clock_s or 0.0:>8.1f}  {','.join(run.verdict_samples)}"
        )
    lowest = min((r for r in manifest.runs if r.confidence is not None),
                 key=lambda r: r.confidence, default=None)
    if lowest:
        print(f"\n§7 stability re-runs: {lowest.ticker} (lowest confidence "
              f"{lowest.confidence:.2f}) and NFLX (least-tested input).")

    return gate.report()


if __name__ == "__main__":
    sys.exit(main())
