"""The worksheet generator is the thing that makes a re-audit affordable
(§8: any Class A/B/E defect forces all six memos to be re-run and
re-audited), so it gets tested before the battery rather than after — a
broken net is worth discovering now, not after six live runs.

The load-bearing case is Class B, and the fixture is the real one: Phase 7's
MSFT memo said net leverage was "declining" over 0.15x -> 0.18x. Both
numbers had provenance, so exact containment certified them; the
relationship between them was wrong and nothing automated caught it.
"""

from __future__ import annotations

import json

import scripts.build_audit_worksheet as ws


def _memo(**fields) -> dict:
    base = {
        "ticker": "TEST",
        "reasoning": "",
        "bull_case": "",
        "bear_case": "",
        "research_thesis": "",
        "risk_debate_summary": "",
        "technical_signal": "",
        "watch_items": [],
    }
    base.update(fields)
    return base


def test_extracts_the_msft_class_b_sentence():
    memo = _memo(reasoning="Net leverage is declining, from 0.15x to 0.18x [RF487E].")
    rows = ws.rows_for(memo)
    assert len(rows) == 1
    row = rows[0]
    assert row["field"] == "reasoning"
    assert row["has_direction_word"] is True
    assert row["cited"] == "[RF487E]"
    # The auditor's columns start empty on purpose: a blank `defect_class`
    # means "checked and clean" only once a human has been through the row,
    # which is why `recomputed_value` is filled in even when it matches.
    assert row["defect_class"] == ""
    assert row["recomputed_value"] == ""


def test_skips_a_sentence_with_no_relational_marker():
    memo = _memo(reasoning="The company operates three reportable segments.")
    assert ws.rows_for(memo) == []


def test_scans_watch_items_which_are_a_list_not_a_string():
    memo = _memo(watch_items=["Volume surges above 1.1x the 20-day average [RF3755]"])
    rows = ws.rows_for(memo)
    assert len(rows) == 1
    assert rows[0]["field"] == "watch_items"


def test_each_watch_item_is_its_own_row():
    """Caught on the real NFLX memo: joining the list and sentence-splitting
    produced ONE row holding all five items, because watch_items entries end
    without terminal punctuation and the splitter had nothing to split on.
    Five unrelated assertions in one row is not an auditable unit."""
    memo = _memo(watch_items=[
        "Volume surges above 1.1x the 20-day average [RF3755]",
        "Price closes above 88.13 on conviction volume [RFC50B]",
        "Price reverses below 82.19, confirming a false signal [RF901B]",
    ])
    rows = ws.rows_for(memo)
    assert len(rows) == 3
    assert [r["cited"] for r in rows] == ["[RF3755]", "[RFC50B]", "[RF901B]"]


def test_flags_a_period_label_for_the_class_d_check():
    memo = _memo(research_thesis="Revenue grew 12.4% in FY2025 versus FY2024.")
    rows = ws.rows_for(memo)
    assert rows[0]["has_period_label"] is True


def test_finds_both_citation_forms_this_pipeline_actually_emits():
    # Not `EV-\d+`. Claims are `[C:id]`, risk factors `[RFxxxx]` with four
    # HEX chars — the shape `_content_id` mints, not the `RF00` of the old
    # positional scheme.
    memo = _memo(reasoning="Margin fell 3.1% [C:acn-margin] under pressure [RFC50B].")
    assert ws.rows_for(memo)[0]["cited"] == "[C:acn-margin] [RFC50B]"


def test_reads_the_memo_back_out_of_a_rendered_markdown_file(tmp_path):
    """The vault writes Markdown, never JSON — the unedited object rides in
    the `## Raw memo` fence. Parsing that rather than the prose keeps the
    worksheet keyed to real field names whichever file it is handed."""
    memo = _memo(reasoning="Leverage rose from 0.15x to 0.18x.")
    md = tmp_path / "TEST-decision.md"
    md.write_text(
        "# TEST — Decision Memo\n\nsome prose 99.9% that is NOT the object\n\n"
        "## Raw memo\n\n```json\n" + json.dumps(memo, indent=2) + "\n```\n"
    )
    assert ws.memo_from(md) == memo
    assert len(ws.rows_for(ws.memo_from(md))) == 1
