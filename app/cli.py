
import asyncio
import logging
import os
from datetime import date
from pathlib import Path

import typer
from dotenv import load_dotenv

from infrastructure.edgar.client import EdgarClient
from infrastructure.edgar.ticker_resolver import TickerResolver
from infrastructure.chunking.section_chunker import chunk_filing
from infrastructure.parsing.filing_parser import parse_filing

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = typer.Typer()

@app.callback()
def main():
    """EDGAR RAG CLI."""
    pass

@app.command()
def fetch(
    ticker: str = typer.Argument(..., help="Stock ticker symbol"),
    form_type: str = typer.Option("10-K", "--type"),
    limit: int = 4,
    since_year: int | None = None,
):
    """Fetch recent filings of a given form type for a ticker."""
    asyncio.run(_fetch(ticker, form_type, limit, since_year))

@app.command(name="inspect-chunks")
def inspect_chunks(
    html_path: Path,
    target_tokens: int = 600,
    overlap_tokens: int = 80,
    preview_chars: int = 400,
    show_first: int = 10,
    json_out: bool = False,
):
    """Parse and chunk one filing; print sections + sample chunks for review."""
    sections = parse_filing(html_path)
    chunks = chunk_filing(sections, target_tokens, overlap_tokens)

    if json_out:
        typer.echo(
            json.dumps(
                [
                    {
                        "section_path": c.section_path,
                        "chunk_index": c.chunk_index,
                        "token_count": c.token_count,
                        "content": c.content,
                    }
                    for c in chunks
                ],
                indent=2,
            )
        )
        return

    typer.echo(f"\nFiling: {html_path}")
    typer.echo(f"Sections detected: {len(sections)}")
    typer.echo(f"Chunks produced:   {len(chunks)}")
    if chunks:
        avg = sum(c.token_count for c in chunks) / len(chunks)
        mx = max(c.token_count for c in chunks)
        mn = min(c.token_count for c in chunks)
        typer.echo(f"Token stats:       min={mn}  avg={avg:.0f}  max={mx}\n")

    typer.echo("=== Section index ===")
    for s in sections:
        typer.echo(f"  [{s.order:>3}] {' > '.join(s.section_path)}")

    typer.echo(f"\n=== First {min(show_first, len(chunks))} chunks ===")
    for c in chunks[:show_first]:
        path = " > ".join(c.section_path)
        typer.echo(f"\n--- Chunk {c.chunk_index:04d}  ({c.token_count} tokens) ---")
        typer.echo(f"section: {path}")
        typer.echo("-" * 60)
        preview = c.content[:preview_chars]
        if len(c.content) > preview_chars:
            preview += "…"
        typer.echo(preview)
        

async def _fetch(ticker: str, form_type: str, limit: int, since_year: int | None) -> None:
    user_agent = os.environ["EDGAR_USER_AGENT"]  # "Wilson Ting wilson@example.com"
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