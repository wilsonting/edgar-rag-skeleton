from datetime import date, datetime
from pydantic import BaseModel, ConfigDict

from .values import AccessionNumber, FilingStatus


class Filing(BaseModel):
    """A single SEC filing (10-K, 10-Q, 8-K, ...)."""
    model_config = ConfigDict(use_enum_values=False)

    id: int | None = None
    security_id: int
    filing_type: str          # "10-K", "10-Q", "8-K"
    filed_date: date
    period_of_report: date | None = None
    accession_number: AccessionNumber
    status: FilingStatus = FilingStatus.DISCOVERED
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def transition_to(self, new_status: FilingStatus) -> None:
        """Move the filing through the state machine."""
        if self.status == FilingStatus.FAILED and new_status != FilingStatus.DISCOVERED:
            # Allow restart from failed by going back to discovered, nothing else
            raise ValueError(
                f"Cannot transition from FAILED to {new_status}; reset to DISCOVERED first"
            )
        self.status = new_status
        if new_status != FilingStatus.FAILED:
            self.error_message = None

    def fail(self, message: str) -> None:
        self.status = FilingStatus.FAILED
        self.error_message = message