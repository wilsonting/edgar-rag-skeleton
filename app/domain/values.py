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
    METRICS_EXTRACTED = "metrics_extracted"
    
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

# First character alphanumeric: the ticker becomes a vault directory name
# (researcher._save_output), and the old pattern `[A-Z0-9.\-]{1,10}` accepted
# ".." — a path-traversal segment. Class shares like BRK.B and BF-B still pass.
_TICKER_RE = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,9}")


def _validate_ticker(v: str) -> str:
    s = str(v).strip().upper()
    if not _TICKER_RE.fullmatch(s):
        raise ValueError(f"Invalid ticker: {v!r}")
    return s


def normalize_ticker(v: str) -> str:
    """Strip and upper-case a ticker, or raise ValueError if it cannot be
    one. The single rule every entry point (HTTP, CLI) and the vault writer
    apply."""
    return _validate_ticker(v)

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