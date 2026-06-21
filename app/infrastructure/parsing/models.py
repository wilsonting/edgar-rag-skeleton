from dataclasses import dataclass

@dataclass(frozen=True)
class ParsedSection:
    """A section extracted from a filing, ready for chunking."""
    section_path: list[str]   # e.g. ["Part I", "Item 1A", "Risk Factors"]
    order: int                # position within the document
    content: str              # cleaned plain text