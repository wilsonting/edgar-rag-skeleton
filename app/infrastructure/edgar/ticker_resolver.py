import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
REFRESH_AFTER = timedelta(days=7)

class TickerResolver:
    """
    Resolves ticker -> CIK using SEC's published map.
    Caches the map locally and refreshes weekly.
    """

    def __init__(self, user_agent: str, cache_path: Path) -> None:
        self.user_agent = user_agent
        self.cache_path = cache_path
        self._index: dict[str, str] | None = None # ticker (upper) -> cik (padded)

    async def _load(self) -> dict[str, str]:
        if self._index is not None:
            return self._index

        if self._needs_refresh():
            await self._download()

    def _need_refresh(self) -> bool:
        if not self.cache_path.exists():
            return True
        age =  datetime.now() -  datetime.fromtimestamp(self.cache_path.stat().st_mtime)
        return age > REFRESH_AFTER

    async def _download(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(
            headers={"User-Agent": self.user_agent}, timeout=30.0
        ) as client:
            resp = await client.get(TICKER_MAP_URL)
            resp.raise_for_status()
            self.cache_path.write_bytes(resp.content)
            logger.info("Refreshed ticker map: %s", self.cache_path)

    async def resolve(self, ticker: str) -> str | None:
        """Return zero-padded 10-digit CIK, or None if ticker unknown."""
        index = await self._load()
        return index.get(ticker.upper())