"""
A memo states its verdict twice — Executive Summary and Assessment — and
nothing reconciled them.

Audited over the 34 memos in the vault on 2026-09-12, 7 disagree with
themselves. The fixtures below are modelled on those, one per shape found:

  FIG  2026-0824  summary MIXED, Assessment IMPAIRED
  NFLX 2026-0827  summary CLEAN, Assessment declares nothing
  AVGO 2026-0827  summary MIXED, Assessment declares nothing
  FIG  2026-0827  item 10(a) a Data Gap, MIXED assigned anyway
  NFLX 2026-0829  summary IMPAIRED, no Assessment section at all
  ACN  2026-0912  item 10(a) a Data Gap, IMPAIRED assigned anyway
  ACN  2026-0912  summary IMPAIRED vs Assessment INSUFFICIENT_EVIDENCE,
                  whose own stated count clears the gate

The oldest is 2026-08-24, so this long predates the retrieval work of
2026-09-12.
"""

from __future__ import annotations

from app.application.memo_verifier import verify_memo
from app.application.verdict_consistency import check_verdict


def _memo(summary: str, assessment: str | None, body: str = "") -> str:
    parts = [
        "# ACN — Research Memo", "", "## Executive Summary",
        f"- {summary}", "",
        "## 1. Free Cash Flow Trend", "FCF rose.", "",
        body or "## Data Gaps\n- **Item 5:** segment detail missing.\n",
    ]
    if assessment is not None:
        parts += ["", "## Assessment", assessment]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Memos that agree with themselves
# ---------------------------------------------------------------------------

def test_a_consistent_memo_is_not_flagged():
    memo = _memo(
        "**Assessment: MIXED, 2 red flag(s)** — two thresholds tripped.",
        "2 of 12 items fully gapped; item 10(a) is answered.\n\n"
        "**Verdict: MIXED**",
    )
    assert check_verdict(memo) == []


def test_naming_the_other_tiers_while_justifying_one_is_not_a_contradiction():
    """The rubric is quoted in the justification of almost every real memo:
    "MIXED — one or more red flags, neither an IMPAIRED trigger". Reading
    those mentions as a second verdict reports a contradiction in a memo that
    is perfectly consistent — it fired on 6 of 34 before being fixed."""
    memo = _memo(
        "**Assessment: MIXED, 2 red flag(s)** — two thresholds tripped.",
        "1 of 12 items fully gapped; item 10(a) is answered.\n\n"
        "Earnings quality tier: MIXED — one or more red flags, neither an "
        "IMPAIRED trigger (no material weakness, and not CLEAN since flags "
        "exist).",
    )
    assert check_verdict(memo) == []


def test_a_verdict_headed_section_is_accepted():
    """One memo headed it "## Verdict" rather than "## Assessment"."""
    memo = _memo(
        "**Assessment: CLEAN** — nothing tripped.", None,
        "## Data Gaps\n- none\n\n## Verdict\n**Verdict: CLEAN**\n",
    )
    assert check_verdict(memo) == []


def test_a_memo_truncated_before_the_checklist_ends_is_left_alone():
    """Ending early is a different failure with its own INCOMPLETE marker.
    Reporting it as a verdict disagreement would bury the real problem."""
    memo = "\n".join([
        "# ACN — Research Memo", "", "## Executive Summary",
        "- **Assessment: MIXED, 1 red flag(s)** — cut off here.",
    ])
    assert check_verdict(memo) == []


# ---------------------------------------------------------------------------
# The seven real shapes
# ---------------------------------------------------------------------------

def test_the_two_sections_naming_different_tiers():
    memo = _memo(
        "**Assessment: MIXED, 1 red flag(s)** — one threshold tripped.",
        "1 of 12 items fully gapped; item 10(a) is answered.\n\n"
        "**Verdict: IMPAIRED**",
    )
    problems = check_verdict(memo)
    assert len(problems) == 1
    assert "MIXED" in problems[0] and "IMPAIRED" in problems[0]


def test_an_assessment_that_declares_no_tier():
    memo = _memo(
        "**Assessment: CLEAN** — nothing tripped.",
        "Coverage: 0 of 12 items fully gapped.\n\nRed flags: none.",
    )
    problems = check_verdict(memo)
    assert len(problems) == 1
    assert "never declares a verdict" in problems[0]


def test_no_assessment_section_at_all_after_a_completed_checklist():
    memo = _memo(
        "**Assessment: IMPAIRED, 2 red flag(s)** — two triggers.", None,
        "## Data Gaps\n- **Item 5:** missing.\n\n## Red-flag rubric\nThresholds apply.\n",
    )
    problems = check_verdict(memo)
    assert len(problems) == 1
    assert "no Assessment section" in problems[0]


def test_item_10a_gapped_forces_insufficient_evidence():
    memo = _memo(
        "**Assessment: IMPAIRED, 3 red flag(s)** — three triggers.",
        "Coverage: 0 of 12 items fully gapped. The Item 10(a) sub-part itself "
        "remains a Data Gap.\n\n**Verdict: IMPAIRED**",
    )
    problems = check_verdict(memo)
    assert any("10(a)" in p and "INSUFFICIENT_EVIDENCE" in p for p in problems)


def test_a_clearing_gate_contradicted_by_an_insufficient_evidence_verdict():
    memo = _memo(
        "**Assessment: IMPAIRED, 3 red flag(s)** — three triggers.",
        "The review has 0 of 12 items fully gapped. Item 10(a), the "
        "ICFR/material-weakness question, is answered.\n\n"
        "**Verdict: INSUFFICIENT_EVIDENCE**",
    )
    problems = check_verdict(memo)
    assert any("gate clears" in p for p in problems)
    assert any("sub-part does not gate" in p for p in problems)


def test_too_many_gapped_items_contradicted_by_an_assigned_tier():
    memo = _memo(
        "**Assessment: MIXED, 1 red flag(s)** — one threshold tripped.",
        "7 of 12 items fully gapped; item 10(a) is answered.\n\n"
        "**Verdict: MIXED**",
    )
    problems = check_verdict(memo)
    assert any("above the limit" in p for p in problems)


def test_clean_with_red_flags():
    memo = _memo(
        "**Assessment: CLEAN, 2 red flag(s)** — contradictory on its face.",
        "0 of 12 items fully gapped; item 10(a) is answered.\n\n"
        "**Verdict: CLEAN**",
    )
    assert any("CLEAN means zero" in p for p in check_verdict(memo))


# ---------------------------------------------------------------------------
# Reaching the memo
# ---------------------------------------------------------------------------

_BAD = _memo(
    "**Assessment: MIXED, 1 red flag(s)** — one threshold tripped.",
    "1 of 12 items fully gapped; item 10(a) is answered.\n\n**Verdict: IMPAIRED**",
)


def test_the_warning_is_appended_even_when_every_figure_checks_out():
    """A memo can be perfectly sourced and still contradict itself about what
    it concludes, so this cannot sit behind the provenance check's early
    return."""
    out = verify_memo(_BAD, provenance_corpus="FCF rose.", computed_values=[])
    assert "## Verdict Inconsistency" in out
    assert out.startswith(_BAD), "the memo itself must not be rewritten"


def test_the_warning_is_appended_when_there_is_no_provenance_at_all():
    out = verify_memo(_BAD, provenance_corpus="", computed_values=[])
    assert "## Verdict Inconsistency" in out


def test_a_consistent_memo_gains_nothing():
    good = _memo(
        "**Assessment: MIXED, 1 red flag(s)** — one threshold tripped.",
        "1 of 12 items fully gapped; item 10(a) is answered.\n\n**Verdict: MIXED**",
    )
    assert verify_memo(good, provenance_corpus="FCF rose.", computed_values=[]) == good
