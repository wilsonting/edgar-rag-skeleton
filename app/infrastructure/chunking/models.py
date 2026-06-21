from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkDraft:
    """A chunk ready for embedding. Not yet a persisted Chunk entity."""
    section_path: list[str]
    chunk_index: int          # global index within the filing
    content: str
    token_count: int