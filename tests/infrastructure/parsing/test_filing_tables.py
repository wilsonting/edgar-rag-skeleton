"""
Coverage for rendering a filing's data tables into section content.

Flattened cell by cell, a statement of cash flows reaches the chunker as
"Net income / $ / 7,832,400 / $ / 7,419,197 / ..." — one cell per paragraph,
with nothing tying a figure to its row or its year. Rendering the table one
row per line keeps that association.

Heading detection still runs on the cell-by-cell blocks: collapsing tables
first moved section boundaries in 14 of 44 cached filings, including dropping
a 10-Q's financial statements, so the two are deliberately kept apart.

No network access — the fragments below are hand-built HTML excerpts.
"""

import warnings
from pathlib import Path

from bs4 import XMLParsedAsHTMLWarning

from app.infrastructure.parsing.filing_parser import parse_filing

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def _table(rows: list[list[str]]) -> str:
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table>{body}</table>"


def _parse(tmp_path: Path, html: str, form_type: str = "10-K"):
    path = tmp_path / "filing.htm"
    path.write_text(f"<html><body>{html}</body></html>", encoding="utf-8")
    return parse_filing(path, form_type=form_type)


_CASH_FLOWS = _table(
    [
        ["", "2025", "2024"],
        ["Net income", "$", "7,832,400", "$", "7,419,197"],
        ["Depreciation", "", "2,441,594", "", "2,168,038"],
        ["Other, net", "(", "200,473", ")", "(", "144,920", ")"],
    ]
)


def test_a_data_table_renders_one_row_per_line(tmp_path):
    sections = _parse(
        tmp_path,
        "<p>Item 8. Financial Statements</p><p>Consolidated Cash Flows</p>" + _CASH_FLOWS,
    )
    content = sections[0].content
    assert "2025 | 2024" in content
    assert "Net income | 7,832,400 | 7,419,197" in content
    assert "Depreciation | 2,441,594 | 2,168,038" in content


def test_currency_and_bracket_fragments_are_glued_back_onto_their_numbers(tmp_path):
    sections = _parse(tmp_path, "<p>Item 8. Financial Statements</p>" + _CASH_FLOWS)
    content = sections[0].content
    assert "Other, net | (200,473) | (144,920)" in content
    assert "| $ |" not in content


def test_the_rendered_table_is_one_paragraph(tmp_path):
    # The chunker splits paragraphs on blank lines; a table separated that way
    # would be cut back into unlabelled fragments.
    sections = _parse(tmp_path, "<p>Item 8. Financial Statements</p>" + _CASH_FLOWS)
    table_block = [
        p for p in sections[0].content.split("\n\n") if "Net income" in p
    ]
    assert len(table_block) == 1
    assert table_block[0].count("\n") == 3


def test_a_table_of_contents_is_not_treated_as_data(tmp_path):
    toc = _table(
        [
            ["Item 7.", "Management's Discussion and Analysis", "45"],
            ["Item 7A.", "Quantitative and Qualitative Disclosures", "60"],
            ["Item 8.", "Financial Statements and Supplementary Data", "62"],
        ]
    )
    sections = _parse(tmp_path, toc + "<p>Item 1. Business</p><p>We do things.</p>")
    assert all("Item 7. | Management" not in s.content for s in sections)


def test_a_single_row_layout_table_is_not_treated_as_data(tmp_path):
    layout = _table([["Revenue for 2025", "100,000"]])
    sections = _parse(
        tmp_path, "<p>Item 7. MD&amp;A</p>" + layout + "<p>Discussion follows.</p>"
    )
    assert "Revenue for 2025 | 100,000" not in sections[0].content
    assert "Revenue for 2025" in sections[0].content


def test_item_headings_inside_tables_are_still_found(tmp_path):
    # Heading detection reads the cell blocks, not the rendered table.
    sections = _parse(
        tmp_path,
        _table([["Item 1A.", "Risk Factors"]])
        + "<p>Our business faces risks.</p>"
        + "<p>Item 2. Properties</p><p>We lease offices.</p>",
    )
    paths = [" > ".join(s.section_path) for s in sections]
    # The number and the title sit in separate cells, so the title is not
    # picked up — but the heading is still located and still bounds a section.
    assert "Part I > Item 1A" in paths
    assert "Part I > Item 2 > Properties" in paths
    risks = next(s for s in sections if s.section_path[1] == "Item 1A")
    assert "Our business faces risks." in risks.content
