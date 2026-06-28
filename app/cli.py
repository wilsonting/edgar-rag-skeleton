
import asyncio
import logging
import os
from datetime import date
from pathlib import Path

import typer
from dotenv import load_dotenv

from app.infrastructure.edgar.client import EdgarClient
from app.infrastructure.edgar.ticker_resolver import TickerResolver
from app.infrastructure.chunking.section_chunker import chunk_filing
from app.infrastructure.parsing.filing_parser import parse_filing
from app.infrastructure.queries.models import FilingDetail, FilingIssue

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


@app.command(name="smoke-persist")
def smoke_persist(ticker: str = "AAPL"):
    """Smoke test the vertical slice: Insert one ListedSecurity + Filing to verify the repository layer."""
    import asyncio
    asyncio.run(_smoke_persist(ticker))
    
async def _smoke_persist(ticker: str) -> None:
    from datetime import date
    from app.domain.listed_security import ListedSecurity
    from app.domain.filing import Filing
    from app.domain.values import FilingStatus
    from app.infrastructure.repositories.db import init_pool, close_pool
    from app.infrastructure.repositories.listed_security_repo import (
        ListedSecurityRepository,
    )
    from app.infrastructure.repositories.filing_repo import FilingRepository

    await init_pool()
    try:
        sec_repo = ListedSecurityRepository()
        fil_repo = FilingRepository()

        security = ListedSecurity(
            cik="320193", ticker=ticker, exchange="NASDAQ", name="Apple Inc."
        )
        security = await sec_repo.upsert(security)
        typer.echo(f"Saved security: id={security.id} cik={security.cik}")

        filing = Filing(
            security_id=security.id,
            filing_type="10-K",
            filed_date=date(2022, 10, 28),
            period_of_report=date(2022, 9, 24),
            accession_number="0000320193-22-000108",
            status=FilingStatus.DISCOVERED,
        )
        filing = await fil_repo.upsert(filing)
        typer.echo(f"Saved filing: id={filing.id} status={filing.status}")

        await fil_repo.mark_status(filing.id, FilingStatus.DOWNLOADED)
        typer.echo("Transitioned to DOWNLOADED")

        roundtrip = await fil_repo.get_by_accession(filing.accession_number)
        typer.echo(f"Roundtrip: {roundtrip}")
    finally:
        await close_pool()

@app.command(name="ingest")
def ingest_cmd(
    ticker: str,
    form_type: str = typer.Option("10-K", "--type"),
    limit: int = typer.Option(4, "--limit"),
    since_year: int | None = typer.Option(None, "--since"),
):
    """Run the full ingestion pipeline for one ticker."""
    asyncio.run(_ingest(ticker, form_type, limit, since_year))

@app.command(name="corpus-status")
def corpus_status_cmd(
    ticker: str | None = typer.Option(None, "--ticker", "-t"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Print a summary of what's actually in the corpus."""
    asyncio.run(_corpus_status(ticker, verbose))

async def _corpus_status(ticker: str | None, verbose: bool) -> None:
    from app.infrastructure.repositories.db import init_pool, close_pool
    from app.infrastructure.queries.corpus_status import CorpusStatusQuery

    await init_pool()
    try:
        query = CorpusStatusQuery()
        summary = await query.summary(ticker)
        if not summary:
            typer.echo("Corpus is empty.")
            return

        _print_summary_table(summary)

        issues = await query.issues(ticker)
        if issues:
            _print_issues(issues)
        else:
            typer.echo("\nāœ“ No stuck or failed filings.")

        if verbose:
            details = await query.per_filing(ticker)
            _print_per_filing(details)
    finally:
        await close_pool()

def _print_summary_table(rows: list[dict]) -> None:
    from datetime import datetime
    typer.echo(f"\nCorpus status — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    header = f"{'ticker':<8} {'filings':>7} {'earliest':>12} {'latest':>12} " \
             f"{'embedded':>9} {'partial':>8} {'failed':>7} {'chunks':>9}"
    typer.echo(header)
    typer.echo("─" * len(header))

    total_filings = total_chunks = 0
    for r in rows:
        line = (
            f"{r.ticker:<8} "
            f"{r.filings:>7} "
            f"{str(r.earliest or '—'):>12} "
            f"{str(r.latest or '—'):>12} "
            f"{r.embedded:>9} "
            f"{r.partial:>8} "
            f"{r.failed:>7} "
            f"{r.chunks:>9,}"
        )
        typer.echo(line)
        total_filings += r.filings
        total_chunks += r.chunks

    typer.echo("─" * len(header))
    typer.echo(
        f"Total: {len(rows)} securities, {total_filings} filings, "
        f"{total_chunks:,} chunks"
    )


def _print_issues(issues: list[FilingIssue]) -> None:
    typer.echo("\n⚠ļø  Issues:")
    for i in issues:
        age = ""
        if i.updated_at:
            from datetime import datetime, timezone
            delta = datetime.now(timezone.utc) - i.updated_at
            age = f" ({delta.days}d ago)" if delta.days > 0 else f" ({delta.seconds // 3600}h ago)"

        line = (
            f"  - {i.ticker} {i.filing_type} "
            f"({i.accession_number}): {i.status}{age}"
        )
        if i.error_message:
            line += f"\n    → {i.error_message}"
        typer.echo(line)


def _print_per_filing(rows: list[FilingDetail]) -> None:
    typer.echo("\nPer-filing breakdown:\n")
    header = f"{'ticker':<8} {'accession':<24} {'type':<6} {'filed':<12} " \
             f"{'status':<11} {'chunks':>7} {'emb':>5}"
    typer.echo(header)
    typer.echo("─" * len(header))

    for r in rows:
        emb_pct = ""
        if r.chunks:
            pct = 100 * r.embedded / r.chunks
            emb_pct = f"{pct:.0f}%"
        typer.echo(
            f"{r.ticker:<8} "
            f"{r.accession_number:<24} "
            f"{r.filing_type:<6} "
            f"{str(r.filed_date):<12} "
            f"{r.status:<11} "
            f"{r.chunks:>7,} "
            f"{emb_pct:>5}"
        )

async def _ingest(
    ticker: str, form_type: str, limit: int, since_year: int | None
) -> None:
    import os
    from datetime import date
    from app.infrastructure.edgar.client import EdgarClient
    from app.infrastructure.edgar.ticker_resolver import TickerResolver
    from app.infrastructure.repositories.db import init_pool, close_pool
    from app.infrastructure.repositories.listed_security_repo import (
        ListedSecurityRepository,
    )
    from app.infrastructure.repositories.filing_repo import FilingRepository
    from app.infrastructure.repositories.document_repo import DocumentRepository
    from app.infrastructure.repositories.section_repo import SectionRepository
    from app.infrastructure.repositories.chunk_repo import ChunkRepository
    from app.application.embedding_service import EmbeddingService
    from app.application.ingestion_service import IngestionService

    user_agent = os.environ["EDGAR_USER_AGENT"]
    cache_root = Path(os.environ.get("EDGAR_CACHE_DIR", "./data/edgar-cache"))

    await init_pool()
    try:
        async with EdgarClient(user_agent, cache_root / "filings") as edgar:
            resolver = TickerResolver(
                user_agent, cache_root / "company_tickers.json"
            )
            embedder = EmbeddingService()

            service = IngestionService(
                edgar_client=edgar,
                ticker_resolver=resolver,
                embedding_service=embedder,
                security_repo=ListedSecurityRepository(),
                filing_repo=FilingRepository(),
                document_repo=DocumentRepository(),
                section_repo=SectionRepository(),
                chunk_repo=ChunkRepository(),
            )

            since = date(since_year, 1, 1) if since_year else None
            await service.ingest_security(
                ticker=ticker,
                form_types=[form_type],
                limit=limit,
                since=since,
            )
    finally:
        await close_pool()

if __name__ == "__main__":
    app()