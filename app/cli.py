
# Entry point: .env first, before any app import reads its settings.
from app.config import load_env, require_env

load_env()

import asyncio  # noqa: E402
from dataclasses import asdict  # noqa: E402
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path

import typer

from app.application.embedding_service import EmbeddingService
from app.application.extraction_service import MetricsExtractor
from app.application.ingestion_service import IngestionService
from app.application.query_decomposer import QueryDecomposer
from app.application.retrieval_service import RetrievalService
from app.domain.values import FilingStatus
from app.infrastructure.edgar.client import EdgarClient
from app.infrastructure.edgar.ticker_resolver import TickerResolver
from app.infrastructure.chunking.section_chunker import chunk_filing
from app.infrastructure.parsing.filing_parser import parse_filing
from app.infrastructure.queries.models import FilingDetail, FilingIssue
from app.infrastructure.repositories.chunk_repo import ChunkRepository
from app.infrastructure.repositories.db import close_pool, init_pool
from app.infrastructure.repositories.document_repo import DocumentRepository
from app.infrastructure.repositories.filing_repo import FilingRepository
from app.infrastructure.repositories.listed_security_repo import ListedSecurityRepository
from app.application.citations import format_citation_tag
from eval.runner import DEFAULT_MODE
from app.infrastructure.repositories.metrics_repo import (
    FinancialMetricsRow,
    MetricsRepository,
)
from app.infrastructure.repositories.section_repo import SectionRepository
from eval.extraction_report import serialize_extraction_result
from eval.runner import serialize_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = typer.Typer()

# ----------------------- app command  -----------------------

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
    form_type: str = typer.Option("10-K", "--type"),
    target_tokens: int = 600,
    overlap_tokens: int = 80,
    preview_chars: int = 400,
    show_first: int = 10,
    json_out: bool = False,
):
    """Parse and chunk one filing; print sections + sample chunks for review."""
    sections = parse_filing(html_path, form_type=form_type)
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

@app.command(name="smoke-persist")
def smoke_persist(ticker: str = "AAPL"):
    """Smoke test the vertical slice: Insert one ListedSecurity + Filing to verify the repository layer."""
    import asyncio
    asyncio.run(_smoke_persist(ticker))

@app.command(name="ingest")
def ingest_cmd(
    ticker: str,
    form_type: str = typer.Option("10-K", "--type"),
    limit: int = typer.Option(4, "--limit"),
    since_year: int | None = typer.Option(None, "--since"),
    retry_failed: bool = typer.Option(
        False, "--retry-failed", help="Re-run filings a previous ingest marked FAILED."
    ),
):
    """Run the full ingestion pipeline for one ticker."""
    asyncio.run(_ingest(ticker, form_type, limit, since_year, retry_failed))

@app.command(name="corpus-status")
def corpus_status_cmd(
    ticker: str | None = typer.Option(None, "--ticker", "-t"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Print a summary of what's actually in the corpus."""
    asyncio.run(_corpus_status(ticker, verbose))

@app.command(name="eval")
def eval_cmd(
    test_set: Path = Path("eval/test_set.yaml"),
    mode: str = typer.Option(
        DEFAULT_MODE, "--mode",
        help="full (what /ask does) | hybrid (what extraction does) | vector (baseline)",
    ),
):
    """Run the evaluation harness against the current retrieval pipeline.

    Defaults to `full`, the path POST /ask actually takes. The old
    --decompose/--hybrid flags selected two paths nothing in production
    used, and no flag combination could reach the real one.
    """
    asyncio.run(_run_eval(test_set, 10, mode))

@app.command(name="eval-extraction")
def eval_extraction_cmd(
    test_set: Path = Path("eval/extraction_truth.yaml"),
    k: int = typer.Option(5, "--k"),
):
    """Validate extraction output against hand-verified filing numbers."""
    asyncio.run(_run_extraction_eval(test_set, k))

@app.command(name="extract-metrics")
def extract_metrics_cmd(
    ticker: str = typer.Argument(..., help="Ticker to extract metrics for e.g. FIG"),
    k: int = typer.Option(5, "--k", help="Chunks per metric query"),
):
    """Extract and store financial metrics for all ingested filings of a ticker."""
    asyncio.run(_run_extract_metrics(ticker, k))


# ----------------------- Definitions -----------------------
async def _fetch(ticker: str, form_type: str, limit: int, since_year: int | None) -> None:
    user_agent = require_env("EDGAR_USER_AGENT")   # "Wilson Ting wilson@example.com"
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

def _prune_old_results(prefix: str, keep: int = 10) -> None:
    files = sorted(Path("eval").glob(f"{prefix}-*.json"), key=lambda p: p.name)
    for stale in files[:-keep]:
        stale.unlink()


async def _run_eval(test_set_path: Path, k, mode: str) -> None:
    from eval.runner import run_eval
    from eval.report import report

    await init_pool()
    try:
        results = await run_eval(test_set_path, k, mode)
        print(report(results))

        # Save raw results for diffing across runs
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_path = Path(f"eval/results-{timestamp}.json")
        results_path.write_text(json.dumps(
            [serialize_result(r) for r in results], indent=2
        ))
        print(f"\nRaw results saved to {results_path}")
        _prune_old_results("results")
    finally:
        await close_pool()


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
    ticker: str, form_type: str, limit: int, since_year: int | None,
    retry_failed: bool = False,
) -> None:
    user_agent = require_env("EDGAR_USER_AGENT")
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
                retry_failed=retry_failed,
            )
    finally:
        await close_pool()

async def _run_extract_metrics(ticker: str, k: int) -> None:
    """Extract and store metrics for every embedded filing of one ticker.

    This command had never run. It called `FilingStatus.INGESTED` (no such
    member), `filing_repo.list_by_state` and `.set_state` (neither exists),
    read `f.fiscal_period` off a Filing (which has `period_of_report`), and
    passed the extractor's own model to `MetricsRepository.upsert`, which
    needs the row type. Five failures, none of them caught, because no test
    touched any CLI command.
    """
    await init_pool()
    try:
        embedder = EmbeddingService()
        chunk_repo = ChunkRepository()
        decomposer = QueryDecomposer()
        retrieval = RetrievalService(
            embedding_service=embedder,
            chunk_repo=chunk_repo,
            decomposer=decomposer,
        )
        extractor = MetricsExtractor()
        metrics_repo = MetricsRepository()
        filing_repo = FilingRepository()

        # EMBEDDED is where ingestion leaves a filing that finished; a
        # filing already at METRICS_EXTRACTED is re-done on request, since
        # the upsert refreshes rather than duplicates.
        filings = await filing_repo.list_by_ticker_and_status(
            ticker, [FilingStatus.EMBEDDED, FilingStatus.METRICS_EXTRACTED]
        )
        if not filings:
            typer.echo(
                f"No embedded filings found for {ticker.upper()} — "
                f"run `ingest` first, or check `corpus-status {ticker.upper()}`."
            )
            raise typer.Exit(1)

        extracted_count = 0
        for f in filings:
            fiscal_period = f.fiscal_period_label()
            typer.echo(
                f"Extracting {fiscal_period} ({f.filing_type}, {f.filed_date})..."
            )

            chunks = await retrieval.retrieve_for_extraction(ticker, f.filed_date, k=k)
            if not chunks:
                typer.echo("  WARNING: no chunks retrieved — skipping")
                continue

            metrics = await extractor.extract(
                chunks, ticker, fiscal_period, f.filing_type, f.filed_date
            )
            await metrics_repo.upsert(FinancialMetricsRow.from_extraction(
                metrics,
                ticker=ticker.upper(),
                fiscal_period=fiscal_period,
                filing_type=f.filing_type,
                filed_date=f.filed_date,
                source_citations=[format_citation_tag(c) for c in chunks],
            ))
            await filing_repo.mark_status(f.id, FilingStatus.METRICS_EXTRACTED)
            extracted_count += 1

            typer.echo(
                f"  revenue={metrics.revenue}M  "
                f"gross_margin={metrics.gross_margin_pct}%  "
                f"fcf={metrics.free_cash_flow}M  "
                f"ndr={metrics.net_dollar_retention}  "
                f"conf={metrics.extraction_confidence}"
            )

        typer.echo(
            f"\nDone. {extracted_count} of {len(filings)} filing(s) "
            f"extracted for {ticker.upper()}."
        )
    finally:
        await close_pool()


# async wrapper — matches _run_eval structure
async def _run_extraction_eval(test_set_path: Path, k: int) -> None:
    from eval.extract_runner import run_extraction_eval
    from eval.extraction_report import extraction_report

    await init_pool()
    try:
        results = await run_extraction_eval(test_set_path, k)
        print(extraction_report(results))

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_path = Path(f"eval/extraction-results-{timestamp}.json")
        results_path.write_text(json.dumps(
            [serialize_extraction_result(r) for r in results], indent=2
        ))
        print(f"\nRaw results saved to {results_path}")
        _prune_old_results("extraction-results")
    finally:
        await close_pool()

if __name__ == "__main__":
    app()