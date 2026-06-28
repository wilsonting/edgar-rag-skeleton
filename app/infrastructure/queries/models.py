from dataclasses import dataclass
from datetime import date, datetime

@dataclass(frozen=True)
class CorpusSummaryRow:
    ticker: str
    filings: int
    earliest: date | None
    latest: date | None
    embedded: int
    partial: int
    failed: int
    chunks: int
    embedded_chunks: int

@dataclass(frozen=True)
class FilingIssue:
    ticker: str
    accession_number: str
    filing_type: str
    filed_date: date
    status: str
    error_message: str | None
    updated_at: datetime | None
    

@dataclass(frozen=True)
class FilingDetail:
    ticker: str
    accession_number: str
    filing_type: str
    filed_date: date
    status: str
    chunks: int
    embedded: int

