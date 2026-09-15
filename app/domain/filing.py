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

    def fiscal_period_label(self) -> str:
        """A stable, distinct period label for this filing.

        `financial_metrics` is keyed on (ticker, filing_type, fiscal_period),
        so this has to be stable across re-extraction and distinct between
        filings — and it has to be derivable, because the CLI extracts every
        filing of a ticker and has no one to ask.

        Annual reports get "FY<year of the period end>", which is the
        filer's own fiscal year whatever month it ends in: ACN's period
        ending 2025-08-31 is FY2025, exactly as ACN labels it.

        Everything else gets the period end date rather than a quarter
        number. A calendar quarter is NOT a fiscal quarter — the period
        ending 2026-02-28 is Q2 for an August year-end filer and Q1 by the
        calendar — and this type does not know the filer's year-end, so
        labelling it "Q1 2026" would be a guess printed as a fact. The date
        is unambiguous and just as distinct.

        Falls back to `filed_date` when the filer reported no period end,
        marked so the two cannot be confused.
        """
        annual = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F"}
        period = self.period_of_report
        if period is None:
            return f"filed {self.filed_date.isoformat()}"
        if self.filing_type in annual:
            return f"FY{period.year}"
        return f"period ending {period.isoformat()}"