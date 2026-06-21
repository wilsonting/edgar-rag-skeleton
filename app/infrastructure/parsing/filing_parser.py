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

# Matches headings like "Item 1.", "Item 1A.", "ITEM 7A —", "Item 7A. Quantitative..."
# Anchored to start of a line/element so we don't match "Item 1" appearing mid-paragraph.
_ITEM_HEADING_RE = re.compile(
    r"^\s*(?:ITEM|Item)\s+(\d{1,2}[A-Za-z]?)\s*[.\-—–:]?\s*(.*?)\s*$"
)

# In a 10-K the table of contents often repeats every item heading. We want
# only the *content* occurrences, which are typically the SECOND time we see
# each item heading walking the document top-to-bottom.
_TOC_REPEAT_THRESHOLD = 2


def parse_filing(html_path: Path) -> list[ParsedSection]:
    """
    Parse one 10-K HTML file into ordered ParsedSection objects.
    """

    logger.info("Parsing %s", html_path)
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    _strip_noise(soup)
    blocks = _flatten_to_text_blocks(soup)
    item_positions = _locate_item_headings(blocks)
    sections = _slice_sections(blocks, item_positions)

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

def _flatten_to_text_blocks(soup: BeautifulSoup) -> list[str]:
    """
    Walk the document and return a list of cleaned text blocks in order.

    A 'block' is roughly a paragraph or heading — text from a <p>, <div>,
    <h*>, <li>, or <td>. Whitespace is normalized. Empty blocks dropped.
    """
    BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "tr", "section"}
    seen_ids: set[int] = set()
    blocks: list[str] = []

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

    return blocks

def _locate_item_headings(blocks: list[str]) -> list[tuple[int, str, str]]:
    """
    Find blocks that look like 'Item N[A]. <title>' headings.
    Returns list of (block_index, item_number, title) in document order.

    Filters out table-of-contents repeats by keeping only the LAST occurrence
    of each item (TOC comes first, content comes later).
    """
    candidates: dict[str, list[tuple[int, str]]] = {}
    for i, block in enumerate(blocks):
        m = _ITEM_HEADING_RE.match(block)
        if not m:
            continue
        item_no = m.group(1).upper()
        # Reject if the "heading" is suspiciously long (real headings are short)
        if len(block) > 200:
            continue
        title = (m.group(2) or "").strip(" .:—–-")
        candidates.setdefault(item_no, []).append((i, title))

    # Keep the LAST occurrence of each item — that's the actual section,
    # earlier ones are typically the TOC.
    located: list[tuple[int, str, str]] = []
    for item_no, occurrences in candidates.items():
        idx, title = occurrences[-1]
        located.append((idx, item_no, title))

    located.sort(key=lambda x: x[0])
    return located

def _slice_sections(
    blocks: list[str],
    item_positions: list[tuple[int, str, str]],
) -> list[ParsedSection]:
    """Take blocks between consecutive Item headings as one section's content."""
    sections: list[ParsedSection] = []
    for order, (start_idx, item_no, title) in enumerate(item_positions):
        end_idx = (
            item_positions[order + 1][0]
            if order + 1 < len(item_positions)
            else len(blocks)
        )
        body_blocks = blocks[start_idx + 1:end_idx]
        content = "\n\n".join(body_blocks).strip()
        if not content:
            continue

        part = _ITEM_TO_PART.get(item_no, "Unknown")
        path = [part, f"Item {item_no}"]
        if title:
            path.append(title)

        sections.append(
            ParsedSection(section_path=path, order=order, content=content)
        )

    return sections

