from __future__ import annotations
import re
from enum import Enum
from typing import Annotated

from pydantic import BeforeValidator, Field

# -------------------- Enums --------------------

class FilingStatus(str, Enum):
    DISCOVERED = "discovered"
    DOWNLOADED = "downloaded"
    PARSED = "parsed"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {FilingStatus.EMBEDDED, FilingStatus.FAILED}


# -------------------- Value objects --------------------

def _validate_cik(v: str | int) -> str:
    """Normalize CIK to a 10-digit zero-padded string."""
    s = str(v).strip()
    if not s.isdigit():
        raise ValueError(f"CIK must be numeric: {v!r}")
    return s.zfill(10)

def _validate_ticker(v: str) -> str:
    s = v.strip().upper()
    if not s or not re.fullmatch(r"[A-Z0-9.\-]{1,10}", s):
        raise ValueError(f"Invalid ticker: {v!r}")
    return s

def _validate_accession(v: str) -> str:
    s = v.strip()
    # SEC format: 10-2-6 digits, separated by dashes
    if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", s):
        raise ValueError(f"Invalid accession number format: {v!r}")
    return s

CIK = Annotated[str, BeforeValidator(_validate_cik)]
Ticker = Annotated[str, BeforeValidator(_validate_ticker)]
AccessionNumber = Annotated[str, BeforeValidator(_validate_accession)]

# SectionPath is just a list of strings, but we name it for clarity
SectionPath = Annotated[list[str], Field(min_length=1)]