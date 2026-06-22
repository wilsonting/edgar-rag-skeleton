from datetime import datetime
from pydantic import BaseModel, ConfigDict

from .values import CIK, Ticker


class ListedSecurity(BaseModel):
    """A publicly listed security tracked in the system."""
    model_config = ConfigDict(frozen=False)  # mutable: status/timestamps change

    id: int | None = None        # None before persistence
    cik: CIK
    ticker: Ticker
    exchange: str | None = None
    name: str
    created_at: datetime | None = None
    updated_at: datetime | None = None