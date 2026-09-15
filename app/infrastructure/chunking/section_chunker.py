import re

import tiktoken
from app.infrastructure.parsing.models import ParsedSection
from .models import ChunkDraft

# cl100k_base matches GPT-4 / text-embedding-3-* tokenization closely enough
# for chunk-size planning purposes.
_ENCODER = tiktoken.get_encoding("cl100k_base")


def chunk_filing(
    sections: list[ParsedSection],
    target_tokens: int = 600,
    overlap_tokens: int = 80,
    min_chunk_tokens: int = 50,
) -> list[ChunkDraft]:
    """
    Produce chunks from parsed sections. Rules:
      1. Chunks never cross section boundaries.
      2. Paragraphs are kept whole when they fit.
      3. Overlap is added between adjacent chunks of the same section.
      4. Very small sections become single small chunks.
    """
    drafts: list[ChunkDraft] = []
    global_index = 0

    for section in sections:
        for chunk_text, token_count in _chunk_one_section(
            section.content, target_tokens, overlap_tokens, min_chunk_tokens
        ):
            drafts.append(
                ChunkDraft(
                    section_path=section.section_path,
                    chunk_index=global_index,
                    content=chunk_text,
                    token_count=token_count,
                )
            )
            global_index += 1

    return drafts


def _chunk_one_section(
    text: str,
    target_tokens: int,
    overlap_tokens: int,
    min_chunk_tokens: int,
) -> list[tuple[str, int]]:
    """Split one section's text into (chunk_text, token_count) pairs."""
    tokens = _ENCODER.encode(text)
    if len(tokens) <= target_tokens:
        if len(tokens) < min_chunk_tokens:
            return []   # drop trivially small sections (headers only, etc.)
        return [(text, len(tokens))]

    # Prefer to split on paragraph boundaries; fall back to token windows.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return _token_window_split(tokens, target_tokens, overlap_tokens)

    chunks: list[tuple[str, int]] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for position, para in enumerate(paragraphs):
        para_tokens = len(_ENCODER.encode(para))

        if para_tokens > target_tokens:
            # Flush buffer first
            if buffer:
                joined = "\n\n".join(buffer)
                chunks.append((joined, buffer_tokens))
                buffer, buffer_tokens = [], 0
            if "\n" in para:
                # A rendered table (the parser emits one row per line). Split
                # it on row boundaries and repeat its headings, rather than
                # cutting mid-row and leaving the numbers unlabelled.
                chunks.extend(
                    _table_split(para, paragraphs[:position], target_tokens)
                )
            else:
                # Then split the oversized paragraph by token window
                chunks.extend(
                    _token_window_split(
                        _ENCODER.encode(para), target_tokens, overlap_tokens
                    )
                )
            continue

        if buffer_tokens + para_tokens > target_tokens:
            joined = "\n\n".join(buffer)
            chunks.append((joined, buffer_tokens))
            # Start new buffer with overlap from the tail of the previous chunk
            buffer, buffer_tokens = _carry_overlap(joined, overlap_tokens)
            buffer.append(para)
            buffer_tokens += para_tokens
        else:
            buffer.append(para)
            buffer_tokens += para_tokens

    if buffer and buffer_tokens >= min_chunk_tokens:
        chunks.append(("\n\n".join(buffer), buffer_tokens))

    return chunks


_NUMERIC_CELL = re.compile(r"^\(?-?[\d,]+(?:\.\d+)?\)?%?$")
_MAX_HEADER_ROWS = 4
_MAX_CONTEXT_PARAS = 2
_MAX_CONTEXT_TOKENS = 40


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.split("|")]


def _is_data_row(row: str) -> bool:
    """A row where a label sits beside at least one number. "2025 | 2024 |
    2023" is not one — its first cell is a number, so it is a column heading."""
    cells = _cells(row)
    return (
        len(cells) >= 2
        and not _NUMERIC_CELL.match(cells[0])
        and any(_NUMERIC_CELL.match(c) for c in cells[1:])
    )


def _split_headings(rows: list[str]) -> tuple[list[str], list[str]]:
    """Separate the column-heading rows from the body.

    Headings are the multi-cell rows before the first data row — the years,
    and any units or currency row above them. Single-cell rows ("CASH FLOWS
    FROM OPERATING ACTIVITIES:") stay in the body: they caption one part of
    the table, so repeating them over the rest would misattribute it.
    """
    headings: list[str] = []
    body: list[str] = []
    seen_data = False
    for row in rows:
        seen_data = seen_data or _is_data_row(row)
        if (
            not seen_data
            and len(headings) < _MAX_HEADER_ROWS
            and len(_cells(row)) >= 2
        ):
            headings.append(row)
        else:
            body.append(row)
    return headings, body


def _table_context(preceding: list[str]) -> list[str]:
    """The one or two short lines before a table — typically its title and its
    units ("(In thousands of U.S. dollars)"). A long paragraph before the
    table is prose, not a caption, so it and anything before it are left out."""
    context: list[str] = []
    for para in reversed(preceding[-_MAX_CONTEXT_PARAS:]):
        if "\n" in para or len(_ENCODER.encode(para)) > _MAX_CONTEXT_TOKENS:
            break
        context.insert(0, para)
    return context


def _table_split(
    para: str, preceding: list[str], target_tokens: int
) -> list[tuple[str, int]]:
    """Split a rendered table on row boundaries, repeating its caption and
    column headings on every piece.

    A token window cuts mid-row and puts the headings in the first chunk only,
    so every later chunk is a column of numbers with no years and no units on
    it. Retrieval then returns figures the answer model cannot attribute; on
    2026-09-11 it reported ACN's operating cash flow as absent from excerpts
    that contained it.
    """
    rows = [r.strip() for r in para.split("\n") if r.strip()]
    headings, body = _split_headings(rows)
    if not body:
        return _token_window_split(_ENCODER.encode(para), target_tokens, 0)

    # The preamble is repeated on every piece, so cap what it may cost.
    preamble = _table_context(preceding) + headings
    while preamble and _count("\n".join(preamble)) > target_tokens // 3:
        preamble.pop(0)
    preamble_tokens = _count("\n".join(preamble)) if preamble else 0

    chunks: list[tuple[str, int]] = []
    piece: list[str] = []
    piece_tokens = 0

    for row in body:
        row_tokens = _count(row)
        if piece and preamble_tokens + piece_tokens + row_tokens > target_tokens:
            text = "\n".join(preamble + piece)
            chunks.append((text, _count(text)))
            piece, piece_tokens = [], 0
        piece.append(row)
        piece_tokens += row_tokens

    if piece:
        text = "\n".join(preamble + piece)
        chunks.append((text, _count(text)))
    return chunks


def _count(text: str) -> int:
    return len(_ENCODER.encode(text))


def _token_window_split(
    tokens: list[int], target_tokens: int, overlap_tokens: int
) -> list[tuple[str, int]]:
    """Sliding token window — last resort for huge unbroken paragraphs."""
    chunks: list[tuple[str, int]] = []
    start = 0
    while start < len(tokens):
        end = min(start + target_tokens, len(tokens))
        slice_ = tokens[start:end]
        chunks.append((_ENCODER.decode(slice_), len(slice_)))
        if end == len(tokens):
            break
        start = end - overlap_tokens
    return chunks


def _carry_overlap(prev_chunk_text: str, overlap_tokens: int) -> tuple[list[str], int]:
    """Take the tail of the previous chunk as overlap into the next buffer."""
    tokens = _ENCODER.encode(prev_chunk_text)
    if len(tokens) <= overlap_tokens:
        return [prev_chunk_text], len(tokens)
    tail = _ENCODER.decode(tokens[-overlap_tokens:])
    return [tail], overlap_tokens