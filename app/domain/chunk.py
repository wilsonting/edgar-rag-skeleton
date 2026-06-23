from datetime import date, datetime
from pydantic import BaseModel, Field

from .values import SectionPath, Ticker
from app.infrastructure.chunking.models import ChunkDraft


class Chunk(BaseModel):
    """A retrievable, embeddable piece of a section."""
    id: int | None = None
    section_id: int
    content: str
    chunk_index: int
    token_count: int
    embedding: list[float] | None = None    # None until embedded

    # Denormalized for fast metadata-filtered retrieval
    ticker: Ticker
    filed_date: date
    filing_type: str
    section_path: SectionPath

    created_at: datetime | None = None

    @property
    def is_embedded(self) -> bool:
        return self.embedding is not None

def chunk_from_draft(
    draft: ChunkDraft,
    section_id: int,
    ticker: str,
    filed_date: date,
    filing_type: str,
) -> Chunk:
    """Lift a chunker output into a persistable domain Chunk."""
    return Chunk(
        section_id=section_id,
        content=draft.content,
        chunk_index=draft.chunk_index,
        token_count=draft.token_count,
        ticker=ticker,
        filed_date=filed_date,
        filing_type=filing_type,
        section_path=draft.section_path,
    )