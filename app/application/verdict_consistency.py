"""Does the memo's headline verdict agree with the section that decides it?

A fundamentals memo states its verdict twice: once as the Executive
Summary's first bullet, and once in the Assessment section that applies the
coverage gate. They are written hundreds of lines apart, and nothing
reconciled them. Audited over the 34 memos in the vault on 2026-09-12, 8
disagreed with themselves, in four ways:

  4  the Assessment section never declares a tier at all — the verdict
     exists only in the Executive Summary, copied from nothing
  2  item 10(a) is a Data Gap, which forces INSUFFICIENT_EVIDENCE, but a
     tier was assigned anyway
  2  the two sections name different tiers outright
  1  the Assessment states "0 of 12 items fully gapped" and "item 10(a) is
     answered" — its own gate clearing — then returns INSUFFICIENT_EVIDENCE

The oldest is from 2026-08-24, so this is not new. Prompt wording alone has
not held it: the instruction to copy the Assessment verdict into the summary
was already there. This finds the disagreement mechanically, and
`verify_memo` appends it to the memo the way it appends unverified figures —
flagged, not hidden, and never silently rewritten, because which of the two
verdicts is the right one is a judgement this code cannot make.
"""

from __future__ import annotations

import re

TIERS = ("CLEAN", "MIXED", "IMPAIRED", "INSUFFICIENT_EVIDENCE")

# The Executive Summary's first bullet: "**Assessment: MIXED, 2 red flag(s)**"
_EXEC_RE = re.compile(r"\*\*Assessment:\s*([A-Z_]+)(?:\s*,\s*(\d+)\s*red flag)?", re.I)

# The Assessment section's own declaration, in any of the forms the memos
# actually use: "**Verdict: X**", "Earnings quality tier: **X**", "Tier: X".
_DECLARED = (
    re.compile(r"\*\*Verdict:\s*([A-Z_]+)\*\*"),
    re.compile(r"Verdict:[*\s]*([A-Z_]+)"),
    re.compile(r"[Ee]arnings[- ]quality tier[:*\s]+([A-Z_]+)"),
    re.compile(r"^[*\s]*Tier[:*\s]+([A-Z_]+)", re.M),
)

_GAP_COUNT = re.compile(r"(\d+)\s+of\s+12\s+(?:top-level\s+)?items fully gapped", re.I)
_TEN_A_ANSWERED = re.compile(r"[Ii]tem 10\(a\)[^.]*\b(?:is|was)\s+answered")
_TEN_A_GAPPED = re.compile(
    r"[Ii]tem 10\(a\)[^.]*\b(?:remains|is|was)\s+(?:itself\s+)?a Data Gap", re.I
)

# More than this many fully-gapped top-level items also forces the verdict.
MAX_GAPPED_ITEMS = 4


def _assessment_section(memo: str) -> str:
    """The section that decides the verdict, or "" if the memo has none.

    "Verdict" as well as "Assessment": one memo in the vault headed it that
    way, and a section named for the thing it decides is not a defect.
    """
    start = re.search(r"^##+\s*(?:Assessment|Verdict)\b", memo, re.M)
    if not start:
        return ""
    rest = memo[start.end():]
    nxt = re.search(r"^##+\s", rest, re.M)
    return rest[: nxt.start()] if nxt else rest


# Sections a memo only reaches by working through the checklist. If one is
# present, the run was not cut off early, so a missing Assessment section is
# a real omission rather than a truncation — which is a different failure,
# with its own INCOMPLETE marker, and not this check's business.
_REACHED_THE_END = re.compile(r"^##+\s*(?:Data Gaps|Red-flag)", re.M)


def _declared_tier(section: str) -> str | None:
    """The tier the section DECLARES.

    Not merely a tier it mentions: the justifications routinely name the
    other tiers ("MIXED — one or more red flags, neither an IMPAIRED
    trigger"), and reading those as a second verdict reports a contradiction
    in a memo that is perfectly consistent.
    """
    for pattern in _DECLARED:
        m = pattern.search(section)
        if m and m.group(1).upper() in TIERS:
            return m.group(1).upper()
    return None


def check_verdict(memo: str) -> list[str]:
    """Every way this memo's two verdicts fail to agree. Empty when fine."""
    exec_match = _EXEC_RE.search(memo)
    exec_tier = exec_match.group(1).upper() if exec_match else None
    if exec_tier not in TIERS:
        exec_tier = None

    section = _assessment_section(memo)
    if not section:
        if exec_tier and _REACHED_THE_END.search(memo):
            return [
                f"The Executive Summary states **{exec_tier}**, but the memo "
                "has no Assessment section — nothing applies the coverage "
                "gate or derives that verdict from the red-flag list."
            ]
        return []   # cut off before the checklist ended: a truncation, not this

    declared = _declared_tier(section)

    gap_match = _GAP_COUNT.search(section) or _GAP_COUNT.search(memo)
    gapped = int(gap_match.group(1)) if gap_match else None
    ten_a_gapped = bool(_TEN_A_GAPPED.search(section) or _TEN_A_GAPPED.search(memo))
    ten_a_answered = (
        bool(_TEN_A_ANSWERED.search(section) or _TEN_A_ANSWERED.search(memo))
        and not ten_a_gapped
    )
    a_tier = ("CLEAN", "MIXED", "IMPAIRED")

    problems: list[str] = []

    if exec_tier and declared is None:
        problems.append(
            f"The Executive Summary states **{exec_tier}**, but the Assessment "
            "section never declares a verdict, so there is nothing it was "
            "copied from."
        )
    elif exec_tier and declared and exec_tier != declared:
        problems.append(
            f"The Executive Summary states **{exec_tier}** and the Assessment "
            f"section states **{declared}**."
        )

    if ten_a_gapped and (exec_tier in a_tier or declared in a_tier):
        problems.append(
            "Item 10(a), the ICFR/material-weakness question, is recorded as a "
            "Data Gap, which requires INSUFFICIENT_EVIDENCE — but the memo "
            f"assigns **{declared or exec_tier}**."
        )

    if gapped is not None:
        verdict = declared or exec_tier
        if verdict == "INSUFFICIENT_EVIDENCE" and gapped <= MAX_GAPPED_ITEMS and ten_a_answered:
            problems.append(
                f"The memo's own coverage gate clears — it states {gapped} of 12 "
                "items fully gapped and item 10(a) answered — yet the verdict is "
                "INSUFFICIENT_EVIDENCE. A gapped sub-part does not gate the "
                "verdict; only a fully gapped top-level item does."
            )
        elif verdict in a_tier and gapped > MAX_GAPPED_ITEMS:
            problems.append(
                f"The memo states {gapped} of 12 items fully gapped, above the "
                f"limit of {MAX_GAPPED_ITEMS}, but assigns **{verdict}** instead "
                "of INSUFFICIENT_EVIDENCE."
            )

    if exec_tier == "CLEAN" and exec_match and exec_match.group(2) and int(exec_match.group(2)):
        problems.append(
            f"The Executive Summary states CLEAN with {exec_match.group(2)} red "
            "flag(s). CLEAN means zero."
        )

    return problems


def verdict_warning(memo: str) -> list[str]:
    """The markdown lines to append, or [] when the memo agrees with itself."""
    problems = check_verdict(memo)
    if not problems:
        return []
    lines = [
        "", "## Verdict Inconsistency", "",
        "This memo states its verdict in two places and they do not agree. "
        "Neither has been changed — resolve it against the Assessment "
        "section's findings before acting on either.", "",
    ]
    lines += [f"- {p}" for p in problems]
    return lines
