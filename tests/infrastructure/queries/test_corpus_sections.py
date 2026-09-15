"""
`check_corpus` tells the agent which section names the `sections` filter
matches.

Given the filter (PR #98) but no list of valid names, the agent guesses
accounting note titles. Measured on ACN 2026-09-12, it passed `sections` on
all 29 ask_edgar calls and **18 of 29 named something no chunk is filed
under** — "Revenue Recognition", "Consolidated Statements of Income",
"Goodwill", "Related Party Transactions". None of those is a section: paths
look like ['Part II', 'Item 9A', 'Controls and Procedures'].
"""

from __future__ import annotations

from app.infrastructure.queries.corpus_status import _item_sort_key, _ITEM_LABEL


def test_item_labels_are_recognised():
    for label in ("Item 1", "Item 1A", "Item 7A", "Item 9A", "Item 16"):
        assert _ITEM_LABEL.match(label), label


def test_note_titles_and_part_headings_are_not_item_labels():
    """These are the names the agent actually guessed, plus the Part and
    title elements that share a section path with the Items."""
    for label in (
        "Revenue Recognition",
        "Consolidated Statements of Income",
        "Goodwill",
        "Related Party Transactions",
        "Part II",
        "Risk Factors",
        "Controls and Procedures",
        "Full Document",
        "Item 4.01",       # an 8-K item number, not a periodic-report Item
    ):
        assert not _ITEM_LABEL.match(label), label


def test_items_sort_in_document_order():
    """Lexicographic order puts Item 10 before Item 2 and Item 1A before
    Item 1 — a list in that order reads as though the corpus is incomplete."""
    labels = ["Item 10", "Item 2", "Item 1A", "Item 1", "Item 16", "Item 7A", "Item 7"]
    assert sorted(labels, key=_item_sort_key) == [
        "Item 1", "Item 1A", "Item 2", "Item 7", "Item 7A", "Item 10", "Item 16",
    ]
