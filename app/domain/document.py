from datetime import datetime
from pydantic import BaseModel


class Document(BaseModel):
    """Metadata + storage pointer for one document within a filing."""
    id: int | None = None
    filing_id: int
    primary_document_name: str    # e.g. "aapl-20240928.htm"
    document_type: str            # e.g. "10-K" (mirrors form type for now)
    original_url: str
    local_path: str
    created_at: datetime | None = None