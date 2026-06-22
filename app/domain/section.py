from datetime import datetime
from pydantic import BaseModel

from .values import SectionPath


class Section(BaseModel):
    """A structural division of a parsed filing."""
    id: int | None = None
    document_id: int
    section_path: SectionPath
    order: int
    content: str
    created_at: datetime | None = None