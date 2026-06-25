from app.infrastructure.repositories.chunk_repo import RetrievedChunk


def format_citation_tag(chunk: RetrievedChunk) -> str:
    """
    Human-readable, unambiguous citation tag for one chunk.

    Format: [TICKER FORM YEAR §Item] e.g. [AAPL 10-K 2025 §Item 1A]
    """
    c = chunk.chunk
    item_label = ""
    for part in c.section_path:
        if part.startswith("Item "):
            item_label = part
            break
    suffix = f" §{item_label}" if item_label else ""
    return f"[{c.ticker} {c.filing_type} {c.filed_date.year}{suffix}]"

def format_context_block(chunks: list[RetrievedChunk]) -> str:
    """Build the context section of the LLM prompt with citation tags."""
    lines = []
    for ch in chunks:
        tag = format_citation_tag(ch)
        section = " > ".join(ch.chunk.section_path)
        lines.append(
            f"{tag} (similarity={ch.similarity:.3f}, section: {section})\n"
            f"{ch.chunk.content}"
        )
    return "\n\n---\n\n".join(lines)