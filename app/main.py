# Entry point (uvicorn app.main:app): .env first, before any app import reads
# its settings. See app/config.py.
from app.config import load_env, require_env

load_env()

import asyncio  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from dataclasses import asdict
import logging
import os
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, Response

import secrets

from fastapi import Depends, Header

from app.domain.token_usage import USAGE_HEADER, TokenUsage, encode_usage_header
from app.domain.values import Ticker, normalize_ticker
from pydantic import BaseModel, ConfigDict, Field
from datetime import date, timedelta
from fastapi.middleware.cors import CORSMiddleware

from app.agent.trading.domain.decision_memo import DecisionMemo
from app.agent.trading.infrastructure.checkpointer import build_checkpointer
from app.agent.trading.infrastructure.graph import build_trading_graph
from app.agent.trading.interface.runner import (
    DEFAULT_MAX_USD,
    DEFAULT_WALL_CLOCK_TIMEOUT_S,
    default_thread_id,
    save_vault_artifacts,
    start_or_resume,
)
from app.agent.researcher import vault_run
from app.application.citations import format_citation_tag
from app.application.citation_verifier import verify_answer
from app.application.embedding_service import EmbeddingService
from app.application.extraction_service import FinancialMetrics, MetricsExtractor
from app.application.ingestion_service import IngestionService
from app.application.query_decomposer import QueryDecomposer
from app.application.retrieval_service import RetrievalService, _fuse_across_queries
from app.application.citations import format_citation_tag

from app.infrastructure.build_info import build_info
from app.infrastructure.llm.models import model_for
from app.infrastructure.edgar.client import EdgarClient, periodic_forms
from app.infrastructure.edgar.ticker_resolver import TickerResolver
from app.infrastructure.queries.corpus_status import CorpusStatusQuery
from app.infrastructure.repositories import metrics_repo
from app.infrastructure.repositories.db import init_pool, close_pool, get_connection
from app.infrastructure.repositories.chunk_repo import (
    ChunkRepository,
    ChunkSearchFilters,
    RetrievedChunk,
)
from app.infrastructure.repositories.document_repo import DocumentRepository
from app.infrastructure.repositories.filing_repo import FilingRepository
from app.infrastructure.repositories.listed_security_repo import ListedSecurityRepository
from app.infrastructure.repositories.section_repo import SectionRepository
from app.infrastructure.repositories.metrics_repo import (
    FinancialMetricsRow,
    MetricsRepository,
)
from app.llm import answer_question


logging.basicConfig(level=logging.INFO)
claude_model = model_for("answer")

@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.infrastructure.build_info import describe
    logging.info("serving code at %s", describe())
    await init_pool()
    # Built once and shared, instead of per request: every /ask and /extract
    # used to construct a fresh OpenAI client and a fresh decomposer client,
    # so no HTTP connection was ever reused. Best-effort — a missing key or
    # provider config leaves them unset and the endpoints build their own,
    # failing then with the same error they always did.
    for name, factory in (("embedder", EmbeddingService), ("decomposer", QueryDecomposer)):
        try:
            setattr(app.state, name, factory())
        except Exception as exc:
            logging.warning("not sharing a %s client: %s", name, exc)
    async with build_checkpointer() as checkpointer:
        app.state.trading_graph = build_trading_graph(checkpointer)
        yield
    await close_pool()


def _embedder() -> EmbeddingService:
    return getattr(app.state, "embedder", None) or EmbeddingService()


def _decomposer() -> QueryDecomposer:
    return getattr(app.state, "decomposer", None) or QueryDecomposer()

app = FastAPI(title="RAG Skeleton", lifespan=lifespan)

# Browsers may call this API only from these origins. It was "*", which let
# ANY page open in the user's browser send requests to localhost:8000 — and
# /ask, /extract, /ingest, /news-assess and /trading/analyze all spend API
# credits. The default is the Vite dev server the removed edgar-ui ran on;
# set CORS_ALLOW_ORIGINS (comma-separated) for anything else, or to "" to
# allow no browser origin at all.
CORS_ALLOW_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ALLOW_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---- Request / response models ----

class AskRequest(BaseModel):
    # A field the server does not understand is a 422, not a shrug. Pydantic's
    # default is to DROP unknown fields, and that is how a 22-hour-stale
    # server accepted `filed_before` on /latest-filings and silently
    # discarded the bound: the caller was told nothing, and a historical run
    # read filings it was not supposed to see. Failing on the first call
    # beats a wrong answer thirty calls later.
    model_config = ConfigDict(extra="forbid")

    question: str
    k: int = 8
    tickers: list[Ticker] | None = None
    filing_types: list[str] | None = None
    filed_after: date | None = None
    filed_before: date | None = None
    section_path_contains: list[str] | None = Field(
        default=None,
        description="e.g. ['Risk Factors'] to restrict to Item 1A sections",
    )

class RetrievedChunkResponse(BaseModel):
    citation: str
    section_path: list[str]
    similarity: float
    # Cosine similarity when the vector search found the chunk; None when only
    # the keyword search did. `similarity` is the fused ranking score.
    vector_similarity: float | None = None
    ticker: str
    filing_type: str
    filed_date: date
    content_preview: str


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    unverified: list[str] = []
    chunks: list[RetrievedChunkResponse]
    dropped_section_filter: list[str] | None = Field(
        default=None,
        description=(
            "Sections that were asked for but match no chunk in this corpus. "
            "When set, the answer was produced WITHOUT the section filter."
        ),
    )

class ExtractRequest(BaseModel):
    # See AskRequest: unknown fields are rejected, not dropped.
    model_config = ConfigDict(extra="forbid")

    ticker: Ticker
    fiscal_period: str          # "Q1 2026" — you supply this, it's not extracted
    filing_type: str            # "10-Q"
    filed_date: date
    filed_after: date | None = None
    filed_before: date | None = None

class FinancialMetricsResponse(BaseModel):
    ticker: str
    fiscal_period: str
    metrics: FinancialMetrics   # the Pydantic model from point 2
    citations: list[str]

class NewsAssessRequest(BaseModel):
    # See AskRequest: unknown fields are rejected, not dropped.
    model_config = ConfigDict(extra="forbid")

    ticker: Ticker
    headline: str

class NewsAssessResponse(BaseModel):
    ticker: str
    headline: str
    assessment: str

class IngestRequest(BaseModel):
    # See AskRequest: unknown fields are rejected, not dropped.
    model_config = ConfigDict(extra="forbid")

    ticker: Ticker
    # None = auto-detect: 10-K for a domestic filer, 20-F for a foreign
    # private issuer (see EdgarClient.default_form_types). Pass explicitly
    # to override, e.g. "10-Q" or "6-K".
    form_type: str | None = None
    limit: int = 3
    since_year: int | None = None
    # Re-run filings a previous ingest marked FAILED (they are skipped
    # otherwise). See IngestionService.ingest_security.
    retry_failed: bool = False

class LatestFilingsRequest(BaseModel):
    # See AskRequest: unknown fields are rejected, not dropped.
    model_config = ConfigDict(extra="forbid")

    ticker: Ticker
    # None = auto-detect the filer's form-type family (see IngestRequest).
    form_types: list[str] | None = None
    since_year: int | None = None
    # Upper bound on the filing date. A historical run must not be shown
    # filings published after the date it is analysing — knowing a 10-K
    # exists is itself lookahead, even before anything is read from it.
    filed_before: date | None = None
    # Narrow the auto-detected family to its PERIODIC members (10-K/10-Q, or
    # 20-F), dropping the event-driven ones. Defaults on because a
    # fundamentals checklist is built from periodic reports and the event
    # filings dominate the list by count -- NFLX returned 44 filings of which
    # 38 were 8-Ks, and every one of them then rode in the agent's context
    # for the rest of the run. Ignored when `form_types` is given explicitly:
    # a caller naming its forms has already said what it wants.
    periodic_only: bool = True

# ---- Auth ----
API_KEY_HEADER = "X-API-Key"


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Opt-in shared secret for the endpoints that spend money.

    Unset `APP_API_KEY` leaves them open — the previous behaviour, and fine
    for a server bound to 127.0.0.1 (uvicorn's default). Set it whenever the
    server is reachable by anything but you: every spending endpoint then
    needs a matching `X-API-Key` header, and the research agent's own tool
    calls send it automatically (app/agent/tools.py reads the same variable).
    Read per request so a test or a restart-free change can move it.
    """
    expected = os.getenv("APP_API_KEY")
    if not expected:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(401, f"missing or invalid {API_KEY_HEADER} header")


SPENDS_MONEY = [Depends(require_api_key)]


# ---- Endpoint ----
def _report_usage(response: Response, *usages: tuple[str, TokenUsage]) -> None:
    """Tell the caller what this request spent, in a header.

    A header and not a body field: the research agent copies tool-result
    bodies verbatim into its provenance corpus, and every numeric guard in
    the trading pipeline does exact containment against that corpus. Four
    token counts added to each retrieval body would be four new numbers that
    could then "back" a figure in a memo. The header leaves the
    agent-visible bytes exactly as they were.

    The caller does the logging, not this server: only it knows the run_id,
    and only its TradingState feeds `check_run_guards`. See
    domain/token_usage.py.

    Each usage is paired with the model that spent it, so the caller can
    price it at that model's rate rather than its own.
    """
    response.headers[USAGE_HEADER] = encode_usage_header(usages)


@app.post("/ask",  response_model=AskResponse, dependencies=SPENDS_MONEY)
async def ask(req: AskRequest, response: Response) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")

    embedder = _embedder()
    chunk_repo = ChunkRepository()
    decomposer = _decomposer()
    retrieval = RetrievalService(
        embedding_service=embedder, 
        chunk_repo=chunk_repo,
        decomposer=decomposer)

    sections = req.section_path_contains
    if sections:
        # An unmatched section name retrieves nothing, and an empty answer
        # reads to the caller as "the filing doesn't say" rather than "you
        # spelled the section wrong". Drop the filter and say so, instead of
        # spending the answer call on no excerpts at all.
        matched = await chunk_repo.sections_with_content(sections, req.tickers)
        if not matched:
            logging.warning(
                "/ask: no chunk is filed under %s; answering without the "
                "section filter", sections,
            )
            sections = None

    filters = ChunkSearchFilters(
        tickers=req.tickers,
        filing_types=req.filing_types,
        filed_after=req.filed_after,
        filed_before=req.filed_before,
        section_path_contains=sections,
    )

    chunks, decomposition = await retrieval.retrieve_full(req.question, k=req.k, filters=filters)
    result = await answer_question(
        question=req.question, 
        chunks=chunks,
        model=claude_model)
    # BOTH calls: the decomposer's rewrite is billed just like the answer.
    _report_usage(
        response, (claude_model, result.usage), (decomposer.model, decomposition.usage)
    )

    report = verify_answer(
        result.answer,
        {c.chunk.id: c.chunk.content for c in chunks},
    )
    logging.info(report.summary())

    return AskResponse(
        answer=result.answer,
        citations=result.citations,
        unverified=[f.value for f in report.unverified], 
        chunks=[
            RetrievedChunkResponse(
                citation=format_citation_tag(c),
                section_path=c.chunk.section_path,
                similarity=c.similarity,
                vector_similarity=c.vector_similarity,
                ticker=c.chunk.ticker,
                filing_type=c.chunk.filing_type,
                filed_date=c.chunk.filed_date,
                content_preview=c.chunk.content[:300] + ("…" if len(c.chunk.content) > 300 else ""),
            )
            for c in chunks
        ],
        dropped_section_filter=(
            req.section_path_contains if sections is None and req.section_path_contains
            else None
        ),
    )


@app.post("/extract", response_model=FinancialMetrics, dependencies=SPENDS_MONEY)
async def extract(req: ExtractRequest, response: Response) -> FinancialMetrics:
    embedder = _embedder()
    chunk_repo = ChunkRepository()
    decomposer = _decomposer()
    retrieval = RetrievalService(
        embedding_service=embedder, 
        chunk_repo=chunk_repo,
        decomposer=decomposer)
    extractor = MetricsExtractor()
    metrics_repo = MetricsRepository()

    # The window defaults to filed_date ± 30 days; a caller that names its
    # own bounds gets them. The extract_metrics tool has always offered
    # filed_after/filed_before to the agent, and they used to be ignored
    # here, so the agent was steering with a control connected to nothing.
    window_start = req.filed_after or req.filed_date - timedelta(days=30)
    window_end = req.filed_before or req.filed_date + timedelta(days=30)
    if window_start > window_end:
        raise HTTPException(
            400, f"filed_after {window_start} is later than filed_before {window_end}"
        )
    chunks = await gather_extraction_chunks(retrieval, req.ticker, window_start, window_end)
    extracted = await extractor.extract(chunks, req.ticker, req.fiscal_period, req.filing_type, req.filed_date)
    # Only the extraction call spends here: gather_extraction_chunks runs
    # fixed queries through retrieve_hybrid, which never calls the decomposer.
    _report_usage(response, (extractor.llm_model, extractor.last_usage))
    # Built through the one constructor both writers share, so the fields
    # the extractor's model does not carry cannot be dropped here either.
    await metrics_repo.upsert(FinancialMetricsRow.from_extraction(
        extracted,
        ticker=req.ticker,
        fiscal_period=req.fiscal_period,
        filing_type=req.filing_type,
        filed_date=req.filed_date,
        source_citations=[format_citation_tag(c) for c in chunks],
    ))
    return extracted

async def gather_extraction_chunks(retrieval: RetrievalService, ticker: str, filed_after, filed_before):
    """The four fixed metric queries, fused into one list.

    Concurrent rather than one at a time, and fused by SUMMING each chunk's
    RRF score across the queries that found it rather than keeping the max —
    both for the reasons `_fuse_across_queries` gives. A chunk the cash-flow
    and income-statement queries both surface is more likely to be the
    statements page than one only a single query reached.
    """
    filters = ChunkSearchFilters(tickers=[ticker], filed_after=filed_after, filed_before=filed_before)
    per_query = await asyncio.gather(*(
        retrieval.retrieve_hybrid(query, k=5, filters=filters)
        for query in RetrievalService.METRIC_QUERIES.values()
    ))
    # No top-k truncation here: the extractor wants every distinct chunk the
    # four queries found, ordered by how much of the set agreed on it.
    return _fuse_across_queries(per_query, k=sum(len(r) for r in per_query))

@app.get("/health")
async def health():
    """What code this process is running.

    Exists because a 22-hour-stale uvicorn served a whole FIG pipeline run
    on 2026-09-13 with no symptom but a date bound that silently did
    nothing. `commit` is snapshotted at import, so it is the RUNNING
    process's commit, not the working tree's.
    """
    return {"status": "ok", **build_info()}


@app.get("/corpus-status")
async def corpus_status_endpoint(ticker: str | None = None, filed_before: date | None = None):
    """`filed_before` bounds every section at the caller's analysis date.

    A historical run could not READ a later filing but could still SEE it:
    a live FIG run at --as-of 2026-03-01 listed two post-cutoff 10-Qs by
    date in its own memo. Knowing a filing exists, and when, is information
    from after the cutoff.
    """
    if ticker is not None:
        try:
            ticker = normalize_ticker(ticker)
        except ValueError as e:
            raise HTTPException(422, str(e))
    query = CorpusStatusQuery()
    summary = await query.summary(ticker, filed_before)
    if not summary:
        return {"summary": [], "issues": [], "per_filing": []}

    issues = await query.issues(ticker, filed_before)
    per_filing = await query.per_filing(ticker, filed_before)

    return {
        "summary": [asdict(row) for row in summary],
        "issues": [asdict(i) for i in issues],
        "per_filing": [asdict(d) for d in per_filing],
        # What `ask_edgar`'s `sections` filter will actually match. Without
        # this the agent guesses note titles, and the filter is dropped.
        "sections_available": await query.item_sections(ticker, filed_before),
    }

@app.post("/ingest", dependencies=SPENDS_MONEY)
async def ingest_endpoint(req: IngestRequest):
    user_agent = require_env("EDGAR_USER_AGENT")
    cache_root = Path(os.environ.get("EDGAR_CACHE_DIR", "./data/edgar-cache"))

    async with EdgarClient(user_agent, cache_root / "filings") as edgar:
        resolver = TickerResolver(user_agent, cache_root / "company_tickers.json")
        embedder = _embedder()
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
        since = date(req.since_year, 1, 1) if req.since_year else None
        await service.ingest_security(
            ticker=req.ticker,
            form_types=[req.form_type] if req.form_type else None,
            limit=req.limit,
            since=since,
            retry_failed=req.retry_failed,
        )

    return {"status": "ok", "ticker": req.ticker, "limit": req.limit}


@app.post("/latest-filings")
async def latest_filings_endpoint(req: LatestFilingsRequest):
    user_agent = require_env("EDGAR_USER_AGENT")
    cache_root = Path(os.environ.get("EDGAR_CACHE_DIR", "./data/edgar-cache"))

    async with EdgarClient(user_agent, cache_root / "filings") as edgar:
        resolver = TickerResolver(user_agent, cache_root / "company_tickers.json")
        cik = await resolver.resolve(req.ticker.upper())
        if not cik:
            raise HTTPException(404, f"Could not resolve ticker {req.ticker}")

        form_types = req.form_types
        if form_types is None:
            form_types = await edgar.default_form_types(cik)
            if req.periodic_only:
                form_types = periodic_forms(form_types)

        since = date(req.since_year, 1, 1) if req.since_year else None
        sec_filings = await edgar.list_filings(
            cik=cik, form_types=form_types, since=since,
        )

    if req.filed_before:
        sec_filings = [f for f in sec_filings if f.filing_date <= req.filed_before]

    accession_numbers = [f.accession_number for f in sec_filings]
    ingested: dict[str, str] = {}
    if accession_numbers:
        async with get_connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT f.accession_number, f.status::text AS status
                FROM filings f
                JOIN listed_securities s ON s.id = f.security_id
                WHERE s.ticker = %s AND f.accession_number = ANY(%s)
                """,
                (req.ticker.upper(), accession_numbers),
            )
            rows = await cur.fetchall()
            ingested = {r["accession_number"]: r["status"] for r in rows}

    filings_list = []
    for f in sec_filings:
        status = ingested.get(f.accession_number)
        filings_list.append({
            "accession_number": f.accession_number,
            "form": f.form,
            "filing_date": f.filing_date.isoformat(),
            "report_date": f.report_date.isoformat() if f.report_date else None,
            "in_corpus": status is not None,
            "corpus_status": status,
        })

    new_filings = [f for f in filings_list if not f["in_corpus"]]
    return {
        "ticker": req.ticker.upper(),
        "form_types_searched": form_types,
        # Echoed so a reader of the agent's trace can see the run was bounded.
        "filed_before": req.filed_before.isoformat() if req.filed_before else None,
        "total_on_sec": len(filings_list),
        "already_ingested": len(filings_list) - len(new_filings),
        "new_filings_count": len(new_filings),
        "filings": filings_list,
    }


@app.post("/news-assess", response_model=NewsAssessResponse, dependencies=SPENDS_MONEY)
async def news_assess(req: NewsAssessRequest) -> NewsAssessResponse:
    if not req.headline.strip():
        raise HTTPException(400, "headline must not be empty")
    if not req.ticker.strip():
        raise HTTPException(400, "ticker must not be empty")

    from app.agent.researcher import run_agent, _build_news_prompt

    ticker = req.ticker.strip().upper()
    prompt = _build_news_prompt(ticker, req.headline)
    task = f"Assess this news for {ticker}:\n\n{req.headline}"
    result, _usage = await run_agent(task, prompt)

    return NewsAssessResponse(
        ticker=ticker,
        headline=req.headline,
        assessment=result,
    )


class TradingAnalysisRequest(BaseModel):
    # See AskRequest: unknown fields are rejected, not dropped.
    model_config = ConfigDict(extra="forbid")

    ticker: Ticker
    thread_id: str | None = None
    # Same defaults as the CLI. `as_of_date` falls back to today HERE, at the
    # boundary — never inside a node (see TradingState.as_of_date). All three
    # apply only to a NEW run: a resume inherits its checkpoint's values.
    as_of_date: date | None = None
    max_usd: float = Field(DEFAULT_MAX_USD, gt=0)
    wall_clock_timeout_s: float = Field(DEFAULT_WALL_CLOCK_TIMEOUT_S, gt=0)

class TradingAnalysisResponse(BaseModel):
    ticker: str
    thread_id: str
    status: Literal["completed", "resumed", "started"]
    # None when the run-level budget or deadline guard stopped the run before
    # synthesis; `run_terminated_by` then says which one.
    decision_memo: DecisionMemo | None = None
    run_terminated_by: str | None = None


@app.post("/trading/analyze", response_model=TradingAnalysisResponse, dependencies=SPENDS_MONEY)
async def trading_analyze(req: TradingAnalysisRequest) -> TradingAnalysisResponse:
    """Run the trading pipeline for one ticker, exactly as the CLI does.

    Blocks for the whole run (minutes). The run lifecycle — initial state,
    budget, recursion limit, stale-resume refusal, run summary, vault
    artifacts — is `interface.runner`'s, shared with the CLI, so the two
    entry points cannot drift apart again.
    """
    if not req.ticker.strip():
        raise HTTPException(400, "ticker must not be empty")

    ticker = req.ticker.strip().upper()
    thread_id = req.thread_id or default_thread_id(ticker)
    as_of = req.as_of_date or date.today()
    graph = app.state.trading_graph

    with vault_run():
        outcome = await start_or_resume(
            graph, ticker, thread_id, as_of,
            max_usd=req.max_usd, wall_clock_timeout_s=req.wall_clock_timeout_s,
        )
        if outcome.status == "refused":
            raise HTTPException(409, outcome.refusal)
        # No terminal capture in a server process — it would interleave every
        # concurrent request's output — so the artifacts carry no run log.
        save_vault_artifacts(outcome.result, run_log="")

    result = outcome.result
    terminated_by = result.get("run_terminated_by")
    return TradingAnalysisResponse(
        ticker=ticker,
        thread_id=thread_id,
        status=outcome.status,
        decision_memo=result.get("decision_memo"),
        run_terminated_by=terminated_by.value if terminated_by else None,
    )
