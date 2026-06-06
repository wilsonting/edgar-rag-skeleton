import asyncio
import logging
import os
from datetime import date
from pathlib import Path

import typer
from dotenv import load_dotenv

from app.infrastructure.edgar.client import EdgarClient
from app.infrastructure.edgar.ticker_resolver import TickerResolver

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = typer.Typer()

@app.command()
def fetch(
    ticker: str,
    form_type: str = typer.Option("10-K", "--type"),
    limit: int = 4,
    since_year: int | None = None,
):
    """Fetch recent filings of a given form type for a ticker."""
    asyncio.run(_fetch(ticker, form_type, limit, since_year))

async def _fetch(ticker: str, form_type: str, limit: int, since_year: int | None) -> None:
    user_agent = os.environ["EDGAR_USER_AGENT"]  
    cache_root = Path(os.environ.get("EDGAR_CACHE_DIR", "./data/edgar-cache"))

    resolver = TickerResolver(user_agent, cache_root / "company_tickers.json")
    cik = await resolver.resolve(ticker)

    if not cik:
        typer.echo(f"Unknown ticker: {ticker}", err=True)
        raise typer.Exit(1)
    typer.echo(f"{ticker.upper()} -> CIK {cik}")

    async with EdgarClient(user_agent, cache_root / "filings") as client:
        since = date(since_year, 1, 1) if since_year else None
        filings = await client.list_filings(cik, form_types=[form_type], since=since)
        filings = filings[:limit]
        typer.echo(f"Found {len(filings)} {form_type} filings:")
        for f in filings:
            typer.echo(f"  {f.filing_date}  {f.accession_number}  {f.primary_document}")

        for f in filings:
            path = await client.download_filing(cik, f)
            typer.echo(f"  cached at {path}  ({path.stat().st_size:,} bytes)")

if __name__ == "__main__":
    app()