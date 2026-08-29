"""Phase 9 §5 — extract the claims a human has to check, into a fixed CSV.

The point is Class B: a relationship stated between two numbers that both
have provenance. `verify_decision_memo` uses exact containment, so both
endpoints pass and the wrong relationship between them is invisible to it —
structurally, not by omission. Nothing automated in this repo can catch that
class, which is why the manual audit exists and why this script exists to
bound it.

Auditing by reading six memos does not survive memo four. Extracting a fixed
row set does: ~10-20 rows per memo, ~90 checks across a battery, each one
recorded whether it was clean or not.

The regex is a NET, NOT A PROOF. It catches sentences carrying a marker — a
percentage, a from/to pair, a multiple, a direction verb, a comparison. A
relational claim phrased without any of those slips through, so §5 pairs
this with one manual skim per memo, and every row added by hand is a pattern
this file should learn.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# Direction verbs, as STEMS not exact forms. "declining" is the word that
# actually appeared in Phase 7's MSFT defect; a list of past-tense forms
# ("declined") misses it, and missing the one confirmed live instance of the
# class the audit exists to bound would be a poor net.
_DIRECTION_VERBS = (
    r"\b(?:grew|grow\w*|rose|ris\w+|fell|fall\w*|declin\w+|increas\w+"
    r"|decreas\w+|improv\w+|deteriorat\w+|expand\w+|contract\w+"
    r"|accelerat\w+|slow\w+|strengthen\w*|weaken\w*|narrow\w*|widen\w*"
    r"|doubl\w+|halv\w+|outpac\w+|lag(?:s|ged|ging)?|up|down)\b"
)

# Sentences asserting a RELATIONSHIP between numbers — the class exact
# containment structurally cannot check, because both endpoints pass it.
RELATIONAL = re.compile(
    r"(\d[\d,.]*\s*%)"                                      # a percentage
    r"|(\bfrom\s+\$?\d[\d,.]*\s*\w*\s+to\s+\$?\d[\d,.]*)"  # from X to Y
    r"|(\b\d[\d,.]*\s*[x\u00d7]\b)"                          # multiples: 0.18x
    r"|" + _DIRECTION_VERBS
    + r"|(\bvs\.?\b|\bversus\b|\bcompared\s+to\b|\brelative\s+to\b)"
    r"|(\b(?:above|below|exceeds?|trails?)\b\s+\$?\d)",   # above 88.13
    re.I,
)

# Direction words — the Class C candidates. Checked against the numbers in
# the SAME sentence, sign independently of magnitude. This is the MSFT
# failure from Phase 7: 0.15x -> 0.18x described as "declining". The number
# was right; the word was wrong; every guard passed.
DIRECTION = re.compile(_DIRECTION_VERBS, re.I)

# Period / entity labels — the Class D surface. FY2024, CY2024 and TTM are
# three different things and a memo has to say which.
LABEL = re.compile(r"\b(FY|CY|Q[1-4]|TTM|H[12])\s?-?\s?\d{0,4}\b|\b(fiscal|calendar)\s+\d{4}\b", re.I)

# Real citation forms in this pipeline — NOT the `EV-\d+` of a generic memo
# schema. `[C:claim-id]` is a debate claim, `[RFxxxx]` a risk-ledger factor
# (4 hex chars; see risk_port._content_id).
CITATION = re.compile(r"\[C:[\w.-]+\]|\[RF[0-9A-Za-z]+\]")

# Every prose field of DecisionMemo. `technical_signal` is included even
# though it is interpreter output rather than debate synthesis: it is the
# densest numeric prose in the memo and a reader treats it identically.
# `data_gaps`, `assumptions` and `evidence` are excluded — the first two are
# Python-assembled, and `evidence` is rendered from resolved references, so
# a defect there is a rendering bug, not a claim to audit.
PROSE_FIELDS = [
    "reasoning",
    "bull_case",
    "bear_case",
    "research_thesis",
    "risk_debate_summary",
    "technical_signal",
    "watch_items",
]

FIELDNAMES = [
    "ticker", "field", "sentence",
    "has_direction_word", "has_period_label", "cited",
    # the auditor fills these in — see §6.2
    "endpoints_found", "recomputed_value", "defect_class", "note",
]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def memo_from(path: Path) -> dict:
    """Accepts either the memo JSON or the rendered Markdown — the Markdown
    embeds the unedited object in its `## Raw memo` fence (see
    decision_memo_port._format_memo_markdown), and the vault only ever
    writes the Markdown. Parsing the embedded JSON rather than the prose
    keeps the worksheet keyed to the real field names either way."""
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    m = re.search(r"## Raw memo\s*\n+```json\n(.*?)\n```", text, re.S)
    if not m:
        raise SystemExit(f"{path}: no '## Raw memo' JSON block — not a rendered memo?")
    return json.loads(m.group(1))


def rows_for(memo: dict) -> list[dict]:
    rows = []
    for field in PROSE_FIELDS:
        value = memo.get(field)
        if value is None:
            continue
        # A list field's elements are separate claims, not one paragraph.
        # Joining them and sentence-splitting the result merges the lot into
        # a single row, because `watch_items` entries carry no terminal
        # punctuation for the splitter to find — and one row holding five
        # unrelated assertions is not auditable.
        units = value if isinstance(value, list) else [str(value)]
        for unit in units:
            for s in sentences(str(unit)):
                if not RELATIONAL.search(s):
                    continue
                rows.append({
                    "ticker": memo["ticker"],
                    "field": field,
                    "sentence": s,
                    "has_direction_word": bool(DIRECTION.search(s)),
                    "has_period_label": bool(LABEL.search(s)),
                    "cited": " ".join(CITATION.findall(s)),
                    "endpoints_found": "",
                    "recomputed_value": "",
                    "defect_class": "",   # blank = checked and clean; A/B/C/D/E
                    "note": "",
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("memos", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for path in args.memos:
        memo = memo_from(path)
        rows = rows_for(memo)
        out = args.out_dir / f"{memo['ticker']}-worksheet.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            w.writerows(rows)
        total += len(rows)
        print(f"{memo['ticker']:6} {len(rows):3} claims -> {out}")
    print(f"\n{total} rows across {len(args.memos)} memo(s). "
          f"A row left untouched is not a clean row — see §6.2 step 5.")


if __name__ == "__main__":
    main()
