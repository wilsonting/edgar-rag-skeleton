# response dataclasses
from dataclasses import dataclass
from datetime import date

@dataclass(frozen=True)
class FilingSummary:
    """One filing as return by the submission endpoint. """
    accession_number: str   # "0000320193-24-000123"
    form: str               # "10-K", "10-Q", "8-K"
    filing_date: date
    filing_data: date | None # period of report
    primary_document: str    # "aapl-20240928.htm"
    report_date: date | None

    @property
    def accession_no_dashes(self) -> str:
        return self.accession_number.replace("-", "")
