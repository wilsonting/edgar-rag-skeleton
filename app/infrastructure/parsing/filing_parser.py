import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

from .models import ParsedSection

logger = logging.getLogger(__name__)

"""
What the code does

1. Strips noise: scripts, styles, hidden divs, and inline XBRL tags (which duplicate numbers and pollute text).
2. Flattens the DOM to ordered text blocks: paragraphs, list items, headings, table cells — each as one string, in document order.
3. Finds Item headings by regex on the block text. Crucially, it keeps only the last occurrence of each item, which filters out the table of contents (TOC headings appear first, real content headings appear later in the doc).
4. Slices content between headings: each Item gets the blocks from its heading up to the next Item's heading.

The TOC-dedup trick is the non-obvious one. Without it you'd get 16 empty "Item 1A" sections pointing at TOC entries. With it, you get one Item 1A section containing actual Risk Factors text.

10-K/10-Q/8-K and 20-F/6-K share the same "Item N[letter]." heading style, so
heading detection is shared; only the item-to-Part bucketing differs by
form_type (20-F has 3 Parts instead of the 10-K's 4). Some foreign private
issuers (e.g. filers using a combined IFRS annual report + 20-F cross-reference
table, like ASML) don't caption their real content with "Item N" headings at
all — only a reference table naming page numbers does — so this parser can't
recover real sections for those filings; it will legitimately find few or no
Item headings rather than mislabeling content.
"""

# Common 10-K Part headings used to bucket items
_ITEM_TO_PART = {
    # Part I
    "1": "Part I", "1A": "Part I", "1B": "Part I", "1C": "Part I",
    "2": "Part I", "3": "Part I", "4": "Part I",
    # Part II
    "5": "Part II", "6": "Part II", "7": "Part II", "7A": "Part II",
    "8": "Part II", "9": "Part II", "9A": "Part II", "9B": "Part II", "9C": "Part II",
    # Part III
    "10": "Part III", "11": "Part III", "12": "Part III",
    "13": "Part III", "14": "Part III",
    # Part IV
    "15": "Part IV", "16": "Part IV",
}

# Form 20-F's Part groupings (confirmed against real filings' own Item/Part
# cross-reference tables): Part I = items 1-12, Part II = items 13-16,
# Part III = items 17-19. Unlike the 10-K, this buckets cleanly by the
# leading item number regardless of letter suffix (e.g. "16G" -> 16 -> Part II),
# so a range lookup is used instead of an exhaustive dict.
_FORM_TYPES_20F = {"20-F", "20-F/A"}


def _item_to_part_20f(item_no: str) -> str:
    m = re.match(r"(\d{1,2})", item_no)
    if not m:
        return "Unknown"
    n = int(m.group(1))
    if 1 <= n <= 12:
        return "Part I"
    if 13 <= n <= 16:
        return "Part II"
    if 17 <= n <= 19:
        return "Part III"
    return "Unknown"


# Matches headings like "Item 1.", "Item 1A.", "ITEM 7A —", "Item 7A. Quantitative..."
# Anchored to start of a line/element so we don't match "Item 1" appearing mid-paragraph.
# This also covers 20-F's top-level items (1-19, including letter-suffixed
# items like "4A" and "16A"-"16K") since real 20-F filings use the same
# "Item N[letter]." caption style as 10-Ks — no separate regex is needed.
_ITEM_HEADING_RE = re.compile(
    r"^\s*(?:ITEM|Item)\s+(\d{1,2}[A-Za-z]?)\s*[.\-—–:]?\s*(.*?)\s*$"
)

# Rejects candidate titles that are checkbox/cross-reference rows rather than
# real headings, e.g. cover-page lines like "Item 17 [ ] Item 18 [ ]" that
# mention a second Item number inline.
_EMBEDDED_ITEM_MENTION_RE = re.compile(r"\bItem\s+\d", re.IGNORECASE)

# Rejects titles that are just glyphs/symbols (checkbox marks) with no
# alphanumeric content at all, e.g. a lone "☐".
_NO_ALNUM_RE = re.compile(r"^[^A-Za-z0-9]+$")

# Periodic reports (unlike event-driven 8-K/6-K filings) are expected to
# spread real content across many items. Some foreign private issuers file a
# combined IFRS annual report + 20-F cross-reference table instead of
# captioning content with "Item N" headings (e.g. ASML) — there, the only
# "Item N" text in the body is unrelated (AGM agenda items, a reference
# table), so heading detection produces a few bogus matches with one
# swallowing most of the document. Below this only shows up as one section
# holding a suspiciously large share of the document's content.
_PERIODIC_FORM_TYPES = {"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A"}
_DOMINANT_SECTION_RATIO = 0.65

# The audited financial statements of an annual report are bound after the
# signature pages, not under the Item that nominally covers them: Item 8 (or
# 20-F Item 18) is a one-line cross-reference and the statements themselves
# follow Item 15/16. Slicing on Item headings alone therefore files a whole
# balance sheet under whichever Item heading happens to precede it — in the
# cached corpus, "Item 16 / Form 10-K Summary" for ACN and NFLX and
# "Item 14 / Principal Accountant Fees" for MSFT, 9 of 16 annual filings.
# The F-pages open with their own index or with the auditor's report, so that
# heading is promoted to a section boundary.
_FPAGES_INDEX_RE = re.compile(
    r"^index to (?:the\s+)?(?:consolidated\s+|combined\s+|condensed\s+)?"
    r"financial statements\s*$",
    re.IGNORECASE,
)
_FPAGES_AUDIT_RE = re.compile(
    r"^report of independent registered public accounting firm\b", re.IGNORECASE
)
_ANNUAL_FORM_TYPES = {"10-K", "10-K/A", "20-F", "20-F/A"}
# The item that owns the audited statements, by form.
_FPAGES_ITEM = {"10-K": "8", "10-K/A": "8", "20-F": "18", "20-F/A": "18"}
_FPAGES_TITLES = ("Financial Statements", "Financial Statements (F-pages)")


def parse_filing(html_path: Path, form_type: str = "10-K") -> list[ParsedSection]:
    """
    Parse one 10-K/10-Q/8-K or 20-F/6-K HTML file into ordered ParsedSection
    objects. `form_type` only affects how items are bucketed into Parts
    (10-K's 4-part scheme vs 20-F's 3-part scheme) — heading detection
    itself is shared across form types.
    """

    logger.info("Parsing %s (form_type=%s)", html_path, form_type)
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    _strip_noise(soup)
    blocks, tables = _flatten_to_text_blocks(soup)
    # Headings are found on the cell-by-cell blocks, exactly as before tables
    # were rendered; only section CONTENT uses the rendered tables. Collapsing
    # tables before heading detection moved section boundaries in 14 of 44
    # cached filings — some better, some dropping a 10-Q's financial
    # statements — so the two are kept apart.
    item_positions = _locate_item_headings(blocks)
    sections = _slice_sections(blocks, item_positions, form_type=form_type, tables=tables)

    # Judge the split before promoting the F-pages, not after: splitting one
    # oversized section in two lowers the largest section's share, and on
    # ASML's 20-F that was enough to let a known-bad split through the check.
    unreliable = False
    if form_type.upper() in _PERIODIC_FORM_TYPES and sections:
        total_chars = sum(len(b) for b in blocks)
        largest = max(len(s.content) for s in sections)
        unreliable = bool(total_chars) and largest / total_chars > _DOMINANT_SECTION_RATIO
        if unreliable:
            logger.warning(
                "Item-heading split looks unreliable for %s (one section holds "
                "%.0f%% of document content) — falling back to a single "
                "whole-document section",
                html_path, 100 * largest / total_chars,
            )
            sections = [
                ParsedSection(
                    section_path=["Unknown", "Full Document"],
                    order=0,
                    content=_join_blocks(blocks, tables).strip(),
                )
            ]

    if not unreliable:
        with_fpages = _add_fpages_heading(blocks, item_positions, form_type)
        if with_fpages is not item_positions:
            sections = _slice_sections(
                blocks, with_fpages, form_type=form_type, tables=tables
            )

    logger.info("Extracted %d non-empty sections", len(sections))
    return sections


def _strip_noise(soup: BeautifulSoup) -> None:
    """Remove tags that pollute extracted text."""
    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()
    # XBRL inline tags carry duplicate numeric content
    for tag in soup.find_all(re.compile(r"^ix:", re.IGNORECASE)):
        tag.unwrap()
    # Hidden elements
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.IGNORECASE)):
        tag.decompose()

def _flatten_to_text_blocks(soup: BeautifulSoup) -> tuple[list[str], list[tuple[int, str] | None]]:
    """
    Walk the document and return a list of cleaned text blocks in order.

    A 'block' is roughly a paragraph or heading — text from a <p>, <div>,
    <h*>, <li>, or <td>. Whitespace is normalized. Empty blocks dropped.

    Also returns, parallel to the blocks, the data table each block belongs
    to — (table id, the table rendered one row per line) — or None. Emitted
    cell by cell, a cash-flow statement reached the chunker as "Net income /
    $ / 7,832,400 / $ / 7,419,197 / ..." one cell per paragraph, with nothing
    tying a number to its row or its year, and the chunker then split it with
    the column headings on one side of the cut and the totals on the other.
    On 2026-09-11 answer models reported ACN's operating-cash-flow rows "not
    in the excerpts" while they were. `_join_blocks` puts the rendered table
    in place of its cells when section content is assembled.
    """
    BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "tr", "section"}
    seen_ids: set[int] = set()
    blocks: list[str] = []
    tables: list[tuple[int, str] | None] = []
    rendered: dict[int, str | None] = {}   # table id -> rendering, None if not a data table

    for tag in soup.find_all(BLOCK_TAGS):
        # Avoid double-counting nested blocks; only emit at the leaf level
        if any(child.name in BLOCK_TAGS for child in tag.find_all()):
            continue
        text = tag.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        # Cheap dedupe by id
        if id(tag) in seen_ids:
            continue
        seen_ids.add(id(tag))
        blocks.append(text)
        tables.append(_data_table_of(tag, rendered))

    return blocks, tables


def _data_table_of(tag, rendered: dict[int, str | None]) -> tuple[int, str] | None:
    """The outermost data table `tag` sits in, rendered — or None."""
    outer = None
    for table in tag.find_parents("table"):
        outer = table
    if outer is None:
        return None
    key = id(outer)
    if key not in rendered:
        rendered[key] = _render_table(outer) if _is_data_table(outer) else None
    return (key, rendered[key]) if rendered[key] else None


def _join_blocks(blocks: list[str], tables: list[tuple[int, str] | None] | None) -> str:
    """Blocks joined as paragraphs, each data table's run of cell blocks
    replaced by the table rendered once, where its first cell was."""
    if tables is None:
        return "\n\n".join(blocks)
    parts: list[str] = []
    emitted: set[int] = set()
    for text, table in zip(blocks, tables):
        if table is None:
            parts.append(text)
        elif table[0] not in emitted:
            emitted.add(table[0])
            parts.append(table[1])
    return "\n\n".join(parts)


_NUMERIC_CELL = re.compile(r"^\(?-?[\d,]+(?:\.\d+)?\)?%?$")
_CURRENCY_ONLY = {"$", "€", "£", "¥"}


def _own_rows(table) -> list:
    """<tr> elements belonging to this table, not to a table nested in it."""
    return [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]


def _row_cells(tr) -> list[str]:
    """A row's non-empty cells, with the currency and bracket fragments that
    filings put in cells of their own glued back onto their numbers:
    "$", "7,832,400" -> "7,832,400"; "(", "4,040,563", ")" -> "(4,040,563)"."""
    cells: list[str] = []
    open_paren = False
    for td in tr.find_all(["td", "th"]):
        if td.find_parent("tr") is not tr:
            continue
        text = re.sub(r"\s+", " ", td.get_text(" ", strip=True)).strip()
        text = re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", text))
        if not text or text in _CURRENCY_ONLY:
            continue
        if text == "(":
            open_paren = True
            continue
        if text in {")", "%", ")%"} and cells:
            cells[-1] += text
            continue
        if open_paren:
            text, open_paren = "(" + text, False
        cells.append(text)
    return cells


def _is_data_table(table) -> bool:
    """Two or more rows where a text label sits beside at least one number.
    A heading laid out in a one-row table does not qualify, and neither does
    a table of contents ("Item 7. | Management's ... | 45")."""
    data_rows = 0
    for tr in _own_rows(table):
        cells = _row_cells(tr)
        if len(cells) >= 2 and not _NUMERIC_CELL.match(cells[0]) and any(
            _NUMERIC_CELL.match(c) for c in cells[1:]
        ):
            if re.match(r"^(part\s+[ivx]+|item\s+\d)", cells[0], re.I):
                return False
            data_rows += 1
    return data_rows >= 2


def _render_table(table) -> str:
    """One line per row, cells joined by " | ". Rows are separated by a
    single newline, so the table stays one paragraph for the chunker, which
    splits paragraphs on blank lines (see section_chunker)."""
    lines = [" | ".join(cells) for cells in (_row_cells(tr) for tr in _own_rows(table)) if cells]
    return "\n".join(lines)

def _locate_item_headings(blocks: list[str]) -> list[tuple[int, str, str]]:
    """
    Find blocks that look like 'Item N[A]. <title>' headings.

    Multiple occurrences of the same Item are common (TOC + content).
    We prefer the occurrence with the most content between it and the
    NEXT Item heading — that's the real section, not a TOC entry.
    """
    candidates: dict[str, list[tuple[int, str]]] = {}
    for i, block in enumerate(blocks):
        m = _ITEM_HEADING_RE.match(block)
        if not m:
            continue
        if len(block) > 200:
            continue
        item_no = m.group(1).upper()
        title = (m.group(2) or "").strip(" .:—–-")
        if title:
            # Checkbox/cross-reference rows, e.g. cover-page lines like
            # "Item 17 [ ] Item 18 [ ]" that mention a second Item inline.
            if _EMBEDDED_ITEM_MENTION_RE.search(title):
                continue
            # Glyph-only titles, e.g. a lone checkbox mark "☐".
            if _NO_ALNUM_RE.match(title):
                continue
            # Real headings start with a capitalized/numeric word; a
            # lowercase-first title means this is a mid-sentence match
            # (e.g. "...as well as NYSE Section 303A.11 requires...").
            if title[0].islower():
                continue
        candidates.setdefault(item_no, []).append((i, title))

    if not candidates:
        return []

    # Flatten all (block_idx, item_no, title) candidates and sort by position
    all_positions = sorted(
        (idx, item_no, title)
        for item_no, occurrences in candidates.items()
        for idx, title in occurrences
    )

    # For each item_no, pick the occurrence with the most blocks before
    # the NEXT heading-of-any-kind. Real content sections have body
    # between them; TOC entries are packed together.
    occurrence_scores: dict[str, list[tuple[int, str, int]]] = {}
    for n, (idx, item_no, title) in enumerate(all_positions):
        next_idx = (
            all_positions[n + 1][0]
            if n + 1 < len(all_positions)
            else len(blocks)
        )
        gap = next_idx - idx - 1  # blocks of body between this heading and next
        occurrence_scores.setdefault(item_no, []).append((idx, title, gap))

    located: list[tuple[int, str, str]] = []
    for item_no, occurrences in occurrence_scores.items():
        # Best occurrence = the one with the largest body gap. Ties (e.g.
        # two equally-short "Not applicable" stub sections) prefer the
        # later position, since TOC entries always precede real content.
        best = max(occurrences, key=lambda x: (x[2], x[0]))
        idx, title, _ = best
        located.append((idx, item_no, title))

    located.sort(key=lambda x: x[0])
    return located

def _fpages_start(blocks: list[str]) -> int | None:
    """Where the audited statements begin: their own index if the filing has
    one (the last such heading — the first is the table-of-contents entry),
    otherwise the auditor's report."""
    index_hits = [i for i, b in enumerate(blocks) if _FPAGES_INDEX_RE.match(b)]
    if index_hits:
        return index_hits[-1]
    audit_hits = [i for i, b in enumerate(blocks) if _FPAGES_AUDIT_RE.match(b)]
    return audit_hits[0] if audit_hits else None


def _span_chars(blocks: list[str], boundaries: list[int], start: int) -> int:
    """Characters of body between the heading at `start` and the next one."""
    later = [b for b in boundaries if b > start]
    end = min(later) if later else len(blocks)
    return sum(len(b) for b in blocks[start + 1:end])


def _add_fpages_heading(
    blocks: list[str],
    item_positions: list[tuple[int, str, str]],
    form_type: str,
) -> list[tuple[int, str, str]]:
    """Promote the start of the F-pages to a section boundary of its own,
    unless they already sit under the Item that covers them."""
    form = form_type.upper()
    if form not in _ANNUAL_FORM_TYPES or not item_positions:
        return item_positions
    start = _fpages_start(blocks)
    if start is None:
        return item_positions

    item_no = _FPAGES_ITEM[form]
    owner = [p for p in item_positions if p[0] < start]
    if not owner or owner[-1][1] == item_no:
        return item_positions   # already filed under Item 8 / Item 18

    # Promote only when the Item that covers the statements is the one-line
    # cross-reference that says they are bound elsewhere. MSFT prints them
    # under Item 8 and also has an "Index to Financial Statements" further
    # down in its exhibit list; there the existing Item 8 is the bigger of
    # the two and the index is not where the statements start.
    existing = next((p[0] for p in item_positions if p[1] == item_no), None)
    boundaries = sorted(p[0] for p in item_positions) + [start]
    if existing is not None and _span_chars(
        blocks, sorted(p[0] for p in item_positions), existing
    ) >= _span_chars(blocks, sorted(boundaries), start):
        return item_positions

    is_20f = form in _FORM_TYPES_20F
    part = _item_to_part_20f(item_no) if is_20f else _ITEM_TO_PART.get(item_no, "Unknown")
    taken = {
        tuple([part, f"Item {no}"] + ([t] if t else []))
        for _, no, t in item_positions
    }
    # Sections are keyed by their path when chunks are attached to them
    # (ingestion_service), so the new one must not duplicate an existing path.
    title = next(
        (t for t in _FPAGES_TITLES if tuple([part, f"Item {item_no}", t]) not in taken),
        None,
    )
    if title is None:
        return item_positions

    logger.info(
        "Filing the F-pages under Item %s; they fell under Item %s",
        item_no,
        owner[-1][1],
    )
    return sorted(item_positions + [(start, item_no, title)])


def _slice_sections(
    blocks: list[str],
    item_positions: list[tuple[int, str, str]],
    form_type: str = "10-K",
    tables: list[tuple[int, str] | None] | None = None,
) -> list[ParsedSection]:
    """Take blocks between consecutive Item headings as one section's content."""
    is_20f = form_type.upper() in _FORM_TYPES_20F
    sections: list[ParsedSection] = []
    for order, (start_idx, item_no, title) in enumerate(item_positions):
        end_idx = (
            item_positions[order + 1][0]
            if order + 1 < len(item_positions)
            else len(blocks)
        )
        body_blocks = blocks[start_idx + 1:end_idx]
        body_tables = tables[start_idx + 1:end_idx] if tables is not None else None
        content = _join_blocks(body_blocks, body_tables).strip()
        if not content:
            continue

        part = _item_to_part_20f(item_no) if is_20f else _ITEM_TO_PART.get(item_no, "Unknown")
        path = [part, f"Item {item_no}"]
        if title:
            path.append(title)

        sections.append(
            ParsedSection(section_path=path, order=order, content=content)
        )

    return sections

