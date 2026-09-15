"""
Regression coverage for filing_parser.py's Item-heading detection.

Validated against real filings before these fixes landed (ASML's and TSM's
FY2025 20-F, AAPL's FY2025 10-K — see PR description for the raw numbers):

  - 20-F's 3-part scheme (Part I = items 1-12, Part II = 13-16, Part III =
    17-19, confirmed against the filers' own Item/Part cross-reference
    tables) instead of reusing 10-K's 4-part dict, which mislabels e.g.
    20-F's Item 13 as "Part III" (it's Part II).
  - Checkbox/cross-reference cover-page lines (e.g. "Item 17 [ ] Item 18 [ ]")
    no longer get matched as real headings — previously one of these swallowed
    293,122 characters of ASML's filing under a bogus "Item 17" section.
  - A heading candidate embedded mid-sentence (e.g. "...Item 16G as well as
    NYSE Section 303A.11 requires...") no longer wins over the real heading —
    previously this mislabeled TSM's real "Item 16G. Corporate Governance"
    section.
  - A dominant-section safety net: some foreign private issuers (e.g. ASML)
    file a combined IFRS annual report + 20-F cross-reference table with no
    "Item N" captions in the body at all — the only "Item N" text is
    unrelated (AGM agenda items). Item-heading detection can't recover real
    sections there, so instead of mislabeling ~83% of the document under a
    bogus Item, parsing falls back to one whole-document section. This only
    applies to periodic reports (10-K/10-Q/20-F) — 8-K/6-K are event-driven
    and legitimately concentrate content under one item.

No real network access — fragments below are small hand-built HTML excerpts,
not full filings.
"""

import warnings
from pathlib import Path

from bs4 import XMLParsedAsHTMLWarning

from app.infrastructure.parsing.filing_parser import (
    _item_to_part_20f,
    _locate_item_headings,
    parse_filing,
)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def _write_html(tmp_path: Path, blocks: list[str]) -> Path:
    body = "\n".join(f"<p>{b}</p>" for b in blocks)
    html_path = tmp_path / "filing.htm"
    html_path.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")
    return html_path


# ---- 20-F Part bucketing ----

def test_20f_part_bucketing_matches_official_form_structure():
    assert _item_to_part_20f("1") == "Part I"
    assert _item_to_part_20f("12D") == "Part I"
    assert _item_to_part_20f("13") == "Part II"
    assert _item_to_part_20f("16K") == "Part II"
    assert _item_to_part_20f("17") == "Part III"
    assert _item_to_part_20f("19") == "Part III"


def test_20f_item_13_is_part_ii_not_part_iii_like_10k():
    """10-K's Item 13 is Part III; 20-F's Item 13 is Part II — reusing the
    10-K dict for 20-F would mislabel this."""
    assert _item_to_part_20f("13") == "Part II"


# ---- Checkbox / false-positive heading rejection ----

def test_checkbox_cover_page_line_is_not_a_heading_candidate():
    blocks = [
        "Item 17 ☐ Item 18 ☐",
        "Cover page filler text about the registrant.",
        "ITEM 17.",
        "Not applicable — see Item 18.",
        "ITEM 18. FINANCIAL STATEMENTS",
        "Total assets were $100 million and total liabilities were $40 million.",
        "ITEM 19. EXHIBITS",
        "Exhibit 1.1 — Articles of Incorporation.",
    ]
    positions = _locate_item_headings(blocks)
    item_nos = [item_no for _, item_no, _ in positions]

    assert "17" in item_nos
    assert "18" in item_nos
    assert "19" in item_nos
    winning_idx_for_17 = next(idx for idx, no, _ in positions if no == "17")
    assert blocks[winning_idx_for_17] != "Item 17 ☐ Item 18 ☐"


def test_mid_sentence_item_mention_does_not_win_over_real_heading(tmp_path):
    blocks = [
        "ITEM 16A. AUDIT COMMITTEE FINANCIAL EXPERT",
        "The board has determined that at least one member qualifies as a financial expert.",
        "Item 16G as well as NYSE Section 303A.11 requires that foreign private "
        "issuers disclose any significant ways in which governance practices differ.",
        "ITEM 16G. CORPORATE GOVERNANCE",
        "Our corporate governance practices differ from NYSE listing standards: "
        "we do not maintain a majority-independent board, consistent with home "
        "country practice under Dutch corporate law.",
        "ITEM 16H. MINE SAFETY DISCLOSURE",
        "Not applicable.",
    ]
    html_path = _write_html(tmp_path, blocks)
    sections = parse_filing(html_path, form_type="20-F")

    section_16g = next(s for s in sections if s.section_path[1] == "Item 16G")
    assert section_16g.section_path[-1] == "CORPORATE GOVERNANCE"
    assert "majority-independent board" in section_16g.content


# ---- TOC vs. content tie-break ----

def test_prefers_titled_later_occurrence_when_toc_and_content_gaps_tie(tmp_path):
    blocks = [
        "ITEM 1.",              # TOC entry, no title
        "TOC filler line.",
        "ITEM 2.",              # TOC entry, no title
        "ITEM 1. REAL TITLE",   # real content, with title
        "Body filler line.",
        "ITEM 2. REAL TITLE 2",  # real content, with title
    ]
    html_path = _write_html(tmp_path, blocks)
    sections = parse_filing(html_path, form_type="20-F")

    item1 = next(s for s in sections if s.section_path[1] == "Item 1")
    assert item1.section_path[-1] == "REAL TITLE"


# ---- Dominant-section fallback (ASML-style filings with no real Item headings) ----

def test_dominant_section_falls_back_to_whole_document_for_20f(tmp_path):
    """Mirrors ASML's filing structure: the only 'Item N' mentions in the
    body are unrelated (AGM agenda items), so detection finds a handful of
    bogus matches, one of which would otherwise swallow almost the entire
    document under a wrong label."""
    filler = " ".join(f"Sentence {i} of unrelated integrated annual report content." for i in range(200))
    blocks = [
        "Item 1",
        "Discussion of the management report and adoption of the financial statements.",
        "Item 2",
        "Discussion of the dividend policy.",
        filler,
    ]
    html_path = _write_html(tmp_path, blocks)
    sections = parse_filing(html_path, form_type="20-F")

    assert len(sections) == 1
    assert sections[0].section_path == ["Unknown", "Full Document"]
    assert "Sentence 0 of unrelated" in sections[0].content


def test_dominant_section_fallback_not_applied_to_event_driven_forms(tmp_path):
    """8-K/6-K are event-driven and legitimately concentrate content under
    one item; the safety net must not suppress that."""
    filler = " ".join(f"Material event detail sentence {i}." for i in range(200))
    blocks = [
        "ITEM 5.02 DEPARTURE OF DIRECTORS",
        filler,
    ]
    html_path = _write_html(tmp_path, blocks)
    sections = parse_filing(html_path, form_type="8-K")

    assert len(sections) == 1
    assert sections[0].section_path != ["Unknown", "Full Document"]


# ---- the F-pages ----
#
# An annual report's audited statements are bound after the signature pages,
# while Item 8 (20-F: Item 18) is a one-line cross-reference. Slicing on Item
# headings alone therefore files a whole balance sheet under whichever Item
# happens to precede it — "Item 16 / Form 10-K Summary" for ACN and NFLX,
# "Item 14 / Principal Accountant Fees" for MSFT, 9 of 16 annual filings in
# the cached corpus.

def _filler(label: str, n: int = 80) -> str:
    return " ".join(f"{label} sentence {i} of the annual report." for i in range(n))


# Enough balancing prose that no single section dominates the document and
# trips the whole-document fallback, which is exercised separately below.
def _stub_10k(tail: list[str]) -> list[str]:
    return [
        "Item 1. Business",
        _filler("Business"),
        "Item 7. Management's Discussion and Analysis",
        _filler("Discussion"),
        "Item 8. Financial Statements and Supplementary Data",
        "See Index to Consolidated Financial Statements.",
        "Item 16. Form 10-K Summary",
        "Not applicable.",
    ] + tail


_STATEMENTS = [
    "Index to Consolidated Financial Statements",
    " ".join(f"Balance sheet line {i} with 1,234,5{i:02d} reported." for i in range(80)),
]


def test_fpages_after_the_last_item_get_their_own_section(tmp_path):
    sections = parse_filing(_write_html(tmp_path, _stub_10k(_STATEMENTS)), form_type="10-K")

    fs = next(s for s in sections if s.section_path[-1] == "Financial Statements")
    assert fs.section_path == ["Part II", "Item 8", "Financial Statements"]
    assert "Balance sheet line 0" in fs.content
    # and they are no longer filed under the Form 10-K Summary
    summary = next(s for s in sections if s.section_path[1] == "Item 16")
    assert "Balance sheet line 0" not in summary.content


def test_fpages_not_promoted_when_item_8_already_holds_them(tmp_path):
    """MSFT prints its statements under Item 8 and has an "Index to Financial
    Statements" further down in its exhibit list; that index is not where the
    statements start, and Item 8 is the bigger of the two."""
    blocks = [
        "Item 1. Business",
        _filler("Business"),
        "Item 8. Financial Statements and Supplementary Data",
        " ".join(f"Balance sheet line {i} with 1,234,5{i:02d} reported." for i in range(80)),
        "Item 15. Exhibits, Financial Statement Schedules",
        "Index to Financial Statements",
        "Schedule II is filed as part of this report.",
    ]
    sections = parse_filing(_write_html(tmp_path, blocks), form_type="10-K")

    assert not any(s.section_path[-1] == "Financial Statements" for s in sections)
    item_8 = next(s for s in sections if s.section_path[1] == "Item 8")
    assert "Balance sheet line 0" in item_8.content


def test_fpages_use_the_auditors_report_when_there_is_no_index(tmp_path):
    tail = [
        "Report of Independent Registered Public Accounting Firm",
        " ".join(f"Audit paragraph {i} on the statements." for i in range(80)),
    ]
    sections = parse_filing(_write_html(tmp_path, _stub_10k(tail)), form_type="10-K")

    fs = next(s for s in sections if s.section_path[-1] == "Financial Statements")
    assert "Audit paragraph 0" in fs.content


def test_fpages_of_a_20f_are_filed_under_item_18(tmp_path):
    blocks = [
        "Item 4. Information on the Company",
        _filler("Company"),
        "Item 5. Operating and Financial Review",
        _filler("Review"),
        "Item 19. Exhibits",
        "The following exhibits are filed.",
        "Index to Consolidated Financial Statements",
        " ".join(f"Balance sheet line {i} with 1,234,5{i:02d} reported." for i in range(80)),
    ]
    sections = parse_filing(_write_html(tmp_path, blocks), form_type="20-F")

    fs = next(s for s in sections if s.section_path[-1] == "Financial Statements")
    assert fs.section_path == ["Part III", "Item 18", "Financial Statements"]


def test_fpages_title_avoids_colliding_with_an_existing_section_path(tmp_path):
    """Chunks are attached to their section by path (ingestion_service), so a
    duplicate path would silently merge two sections."""
    blocks = [
        "Item 1. Business",
        _filler("Business"),
        "Item 7. Management's Discussion and Analysis",
        _filler("Discussion"),
        "Item 8. Financial Statements",
        "See Index to Consolidated Financial Statements.",
        "Item 16. Form 10-K Summary",
        "Not applicable.",
    ] + _STATEMENTS
    sections = parse_filing(_write_html(tmp_path, blocks), form_type="10-K")

    paths = [tuple(s.section_path) for s in sections]
    assert len(paths) == len(set(paths))
    assert ("Part II", "Item 8", "Financial Statements (F-pages)") in paths


def test_fpages_not_promoted_for_quarterly_reports(tmp_path):
    blocks = [
        "Item 2. Management's Discussion and Analysis",
        " ".join(f"Quarterly sentence {i}." for i in range(60)),
        "Report of Independent Registered Public Accounting Firm",
        " ".join(f"Review paragraph {i}." for i in range(40)),
    ]
    sections = parse_filing(_write_html(tmp_path, blocks), form_type="10-Q")

    assert not any(s.section_path[-1] == "Financial Statements" for s in sections)


def test_whole_document_fallback_still_wins_over_the_fpages_split(tmp_path):
    """Splitting one oversized section in two lowers the largest section's
    share of the document, which was enough to let ASML's known-bad 20-F
    split through the reliability check."""
    filler = " ".join(f"Sentence {i} of unrelated integrated annual report content." for i in range(400))
    blocks = [
        "Item 1",
        "Discussion of the management report and adoption of the financial statements.",
        "Item 2",
        "Discussion of the dividend policy.",
        filler,
        "Report of Independent Registered Public Accounting Firm",
        " ".join(f"Audit paragraph {i} on the statements." for i in range(80)),
    ]
    sections = parse_filing(_write_html(tmp_path, blocks), form_type="20-F")

    assert len(sections) == 1
    assert sections[0].section_path == ["Unknown", "Full Document"]


# ---- 10-K regression (unchanged behavior) ----

def test_10k_part_bucketing_still_uses_four_part_scheme(tmp_path):
    blocks = [
        "ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK",
        "Interest rate risk discussion goes here in reasonable detail for the test.",
        "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        "See accompanying financial statements for the fiscal year.",
    ]
    html_path = _write_html(tmp_path, blocks)
    sections = parse_filing(html_path, form_type="10-K")

    section_7a = next(s for s in sections if s.section_path[1] == "Item 7A")
    assert section_7a.section_path[0] == "Part II"
