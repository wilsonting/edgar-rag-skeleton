"""
Coverage for section_chunker's handling of rendered financial tables.

The parser emits a data table as one paragraph, one row per line. A statement
of cash flows is far longer than a chunk, so the chunker has to cut it — and
where it cuts decides whether the numbers stay readable. A sliding token
window cuts mid-row and leaves the column headings in the first piece only,
so every later piece is a column of figures with no years on it. On
2026-09-11 an answer model reported ACN's operating cash flow as absent from
excerpts that contained it.
"""

from app.infrastructure.chunking.section_chunker import (
    _ENCODER,
    _split_headings,
    _table_context,
    chunk_filing,
)
from app.infrastructure.parsing.models import ParsedSection


def _rows(n: int, prefix: str = "Line item") -> list[str]:
    return [f"{prefix} {i} | {i}23,456 | {i}98,765 | {i}11,111" for i in range(n)]


def _section(paragraphs: list[str]) -> ParsedSection:
    return ParsedSection(
        section_path=["Part II", "Item 8", "Financial Statements"],
        order=0,
        content="\n\n".join(paragraphs),
    )


def _chunks(paragraphs: list[str], **kw):
    return chunk_filing([_section(paragraphs)], **kw)


# ---- heading / body separation ----

def test_year_row_is_a_heading_not_a_data_row():
    headings, body = _split_headings(["2025 | 2024 | 2023", "Net income | 1 | 2 | 3"])
    assert headings == ["2025 | 2024 | 2023"]
    assert body == ["Net income | 1 | 2 | 3"]


def test_single_cell_caption_stays_in_the_body():
    # "CASH FLOWS FROM OPERATING ACTIVITIES:" captions one part of the table.
    # Repeated over the financing rows it would misattribute them.
    headings, body = _split_headings(
        ["2025 | 2024", "CASH FLOWS FROM OPERATING ACTIVITIES:", "Net income | 1 | 2"]
    )
    assert headings == ["2025 | 2024"]
    assert body[0] == "CASH FLOWS FROM OPERATING ACTIVITIES:"


def test_headings_stop_at_the_first_data_row():
    headings, body = _split_headings(
        ["(In thousands) | | ", "2025 | 2024", "Revenue | 10 | 20", "2024 | 2023"]
    )
    assert headings == ["(In thousands) | | ", "2025 | 2024"]
    # A heading-shaped row after the data has started belongs to the body.
    assert body == ["Revenue | 10 | 20", "2024 | 2023"]


def test_context_takes_short_lines_and_stops_at_prose():
    prose = "word " * 60
    context = _table_context([prose, "Consolidated Cash Flows Statements", "(In thousands)"])
    assert context == ["Consolidated Cash Flows Statements", "(In thousands)"]
    assert _table_context([prose]) == []


# ---- splitting a real-shaped table ----

def test_every_piece_of_a_split_table_carries_the_headings_and_caption():
    table = "\n".join(["2025 | 2024 | 2023"] + _rows(120))
    chunks = _chunks(
        ["Consolidated Cash Flows Statements", "(In thousands of U.S. dollars)", table]
    )
    table_chunks = [c for c in chunks if "Line item" in c.content]
    assert len(table_chunks) > 1, "table should have been split"
    for c in table_chunks:
        assert c.content.startswith(
            "Consolidated Cash Flows Statements\n(In thousands of U.S. dollars)\n2025 | 2024 | 2023"
        )


def test_rows_are_never_cut_in_half():
    table = "\n".join(["2025 | 2024 | 2023"] + _rows(120))
    chunks = _chunks(["Statement", table])
    for c in chunks:
        for line in c.content.splitlines():
            if line.startswith("Line item"):
                assert line.count("|") == 3, f"row was cut: {line!r}"


def test_every_row_survives_the_split_exactly_once():
    rows = _rows(120)
    chunks = _chunks(["Statement", "\n".join(["2025 | 2024 | 2023"] + rows)])
    body = [
        line
        for c in chunks
        for line in c.content.splitlines()
        if line.startswith("Line item")
    ]
    assert body == rows


def test_pieces_stay_within_the_token_target():
    table = "\n".join(["2025 | 2024 | 2023"] + _rows(120))
    for c in _chunks(["Statement", table], target_tokens=300):
        # One row may push a piece over; more than that means the repeated
        # preamble is eating the budget.
        assert c.token_count <= 300 + 40


def test_prose_still_goes_through_the_token_window():
    # A single oversized paragraph with no newlines is not a table.
    prose = "The registrant reported results. " * 400
    chunks = _chunks([prose])
    assert len(chunks) > 1
    assert all("|" not in c.content for c in chunks)


def test_a_table_that_fits_is_left_whole():
    table = "\n".join(["2025 | 2024 | 2023"] + _rows(5))
    chunks = _chunks(["Statement", table])
    assert len(chunks) == 1
    assert chunks[0].content.count("Line item") == 5


def test_preamble_is_capped_so_it_cannot_crowd_out_the_rows():
    # A pathological table with many wide heading rows and no caption.
    headings = [" | ".join(f"heading cell {i}" for i in range(12)) for _ in range(4)]
    table = "\n".join(headings + _rows(120))
    chunks = _chunks(["Statement", table], target_tokens=200)
    preamble_tokens = len(_ENCODER.encode("\n".join(headings)))
    assert preamble_tokens > 200 // 3
    # Trimmed rather than repeated in full on every piece.
    assert all(c.content.count("heading cell 11") <= 1 for c in chunks)
    assert sum(c.content.count("Line item") for c in chunks) == 120
