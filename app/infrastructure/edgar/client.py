# the HTTP client with rate limiting
import asyncio
import logging
from pathlib import Path
from datetime import date, datetime
from sqlite3 import connect

import httpx
from .models import FilingSummary

logger = logging.getLogger(__name__)

class EdgarRateLimitError(Exception):
    pass

class EdgarClient:
    """
    Minimal SEC EDGAR client.

    SEC requires:
      - User-Agent: "Name email@example.com"
      - <= 10 req/sec
      - https://
    """

    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

    def __init__(
        self,
        user_agent: str,
        cache_dir: Path,
        requests_per_second: float = 8.0 #stay below 10 with margin
    ) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "user_agent must include name and email"
            )
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._semaphore = asyncio.Semaphore(1)
        self._interval = 1.0 / requests_per_second
        self._last_request_at : float = 0.0
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent" : user_agent,
                "Accept-Encoding": "gzip, deflate"
            },
            timeout=httpx.httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "EdgarClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()

    async def _throttled_get(self, url: str ) -> httpx.Response:
        """Single point through which all GETs flow — enforces rate limit."""
        async with self._semaphore:
            elapsed = asyncio.get_event_loop().time() - self._last_request_at
            if elapsed < self._interval:
                await asyncio.sleep(self._interval - elapsed)
            self._last_request_at = asyncio.get_event_loop().time()
        
        logger.debug("GET %s", url)
        resp = await self._client.get(url)
        if resp.status_code == 429:
            raise EdgarRateLimitError(f"Rate limited by SEC: {url}")
        if resp.status_code == 403:
            raise EdgarRateLimitError(
                f"403 Forbidden — likely missing/invalid User-Agent: {url}"
            )
        resp.raise_for_status()
        return resp
    
    async def list_filings(
        self,
        cik: str,
        form_types: list[str] | None = None,
        since: date | None = None
    ) -> list[FilingSummary]:
        """List filings from the submissions endpoint."""
        cik_padded = cik.zfill(10)
        url = self.SUBMISSIONS_URL.format(cik=cik_padded)
        resp = await self._throttled_get(url)
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            return []

        # Submissions JSON uses parallel arrays — same index across all keys
        accessions = recent.get("accessionNumber", [])
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [])

        filings = []
        for i in range(len(accessions)):
            form = forms[i]
            if form_types and form not in form_types:
                continue

            filing_date = datetime.strptime(filing_dates[i], "%Y-%m-%d").date()
            if since and filing_date < since:
                continue

            report_date = None
            if report_dates[i]:
                report_date = datetime.strptime(report_dates[i], "%Y-%m-%d").date()

            filings.append(
                FilingSummary(
                    accession_number=accessions[i],
                    form=form,
                    filing_date=filing_date,
                    report_date=report_date,
                    primary_document=primary_docs[i],
                )
            )

        return filings

    async def download_filing(
        self,
        cik: str,
        filing: FilingSummary,
    ) -> Path:
        """
        Download a filing's primary document. Cached by accession number.
        Returns local file path.
        """
        cik_unpadded = str(int(cik))  # strip leading zeros
        cache_path = (
            self.cache_dir
            / cik_unpadded
            / filing.accession_no_dashes
            / filing.primary_document
        )

        if cache_path.exists():
            logger.debug("Cache hit: %s", cache_path)
            return cache_path

        url = self.ARCHIVES_URL.format(
            cik=cik_unpadded,
            accession=filing.accession_no_dashes,
            document=filing.primary_document,
        )
        resp = await self._throttled_get(url)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(resp.content)
        logger.info("Downloaded %s -> %s", url, cache_path)
        return cache_path