from contextlib import asynccontextmanager
from dataclasses import asdict
import logging
import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response

from app.domain.token_usage import USAGE_HEADER, TokenUsage
from pydantic import BaseModel, Field
from datetime import date, timedelta
from fastapi.middleware.cors import CORSMiddleware

from app.agent.trading.domain.decision_memo import DecisionMemo
from app.agent.trading.infrastructure.checkpointer import build_checkpointer
from app.agent.trading.infrastructure.graph import build_trading_graph
from app.application.citations import format_citation_tag
from app.application.citation_verifier import verify_answer
from app.application.embedding_service import EmbeddingService
from app.application.extraction_service import FinancialMetrics, MetricsExtractor
from app.application.ingestion_service import IngestionService
from app.application.query_decomposer import QueryDecomposer
from app.application.retrieval_service import RetrievalService
from app.application.citations import format_citation_tag

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
from app.infrastructure.repositories.metrics_repo import MetricsRepository
from app.llm import answer_question


load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
claude_model = model_for("answer")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool()
    async with build_checkpointer() as checkpointer:
        app.state.trading_graph = build_trading_graph(checkpointer)
        yield
    await close_pool()

app = FastAPI(title="RAG Skeleton", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your actual origin once you know it
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---- Request / response models ----

class AskRequest(BaseModel):
    question: str
    k: int = 8
    tickers: list[str] | None = None
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
    ticker: str
    filing_type: str
    filed_date: date
    content_preview: str


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    unverified: list[str] = []
    chunks: list[RetrievedChunkResponse]

class ExtractRequest(BaseModel):
    ticker: str
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
    ticker: str
    headline: str

class NewsAssessResponse(BaseModel):
    ticker: str
    headline: str
    assessment: str

class IngestRequest(BaseModel):
    ticker: str
    # None = auto-detect: 10-K for a domestic filer, 20-F for a foreign
    # private issuer (see EdgarClient.default_form_types). Pass explicitly
    # to override, e.g. "10-Q" or "6-K".
    form_type: str | None = None
    limit: int = 3
    since_year: int | None = None

class LatestFilingsRequest(BaseModel):
    ticker: str
    # None = auto-detect the filer's form-type family (see IngestRequest).
    form_types: list[str] | None = None
    since_year: int | None = None
    # Narrow the auto-detected family to its PERIODIC members (10-K/10-Q, or
    # 20-F), dropping the event-driven ones. Defaults on because a
    # fundamentals checklist is built from periodic reports and the event
    # filings dominate the list by count -- NFLX returned 44 filings of which
    # 38 were 8-Ks, and every one of them then rode in the agent's context
    # for the rest of the run. Ignored when `form_types` is given explicitly:
    # a caller naming its forms has already said what it wants.
    periodic_only: bool = True

# ---- Endpoint ----
def _report_usage(response: Response, *usages: TokenUsage) -> None:
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
    """
    total = TokenUsage()
    for usage in usages:
        total = total + usage
    response.headers[USAGE_HEADER] = total.model_dump_json()


@app.post("/ask",  response_model=AskResponse)
async def ask(req: AskRequest, response: Response) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")

    embedder = EmbeddingService()
    chunk_repo = ChunkRepository()
    decomposer = QueryDecomposer()
    retrieval = RetrievalService(
        embedding_service=embedder, 
        chunk_repo=chunk_repo,
        decomposer=decomposer,
        use_hybrid=True)

    filters = ChunkSearchFilters(
        tickers=req.tickers,
        filing_types=req.filing_types,
        filed_after=req.filed_after,
        filed_before=req.filed_before,
        section_path_contains=req.section_path_contains,
    )

    chunks, decomposition = await retrieval.retrieve_full(req.question, k=req.k, filters=filters)
    result = await answer_question(
        question=req.question, 
        chunks=chunks,
        model=claude_model)
    # BOTH calls: the decomposer's rewrite is billed just like the answer.
    _report_usage(response, result.usage, decomposition.usage)

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
                ticker=c.chunk.ticker,
                filing_type=c.chunk.filing_type,
                filed_date=c.chunk.filed_date,
                content_preview=c.chunk.content[:300] + ("…" if len(c.chunk.content) > 300 else ""),
            )
            for c in chunks
        ],
    )


@app.post("/extract", response_model=FinancialMetrics)
async def extract(req: ExtractRequest, response: Response) -> FinancialMetrics:
    embedder = EmbeddingService()
    chunk_repo = ChunkRepository()
    decomposer = QueryDecomposer()
    retrieval = RetrievalService(
        embedding_service=embedder, 
        chunk_repo=chunk_repo,
        decomposer=decomposer,
        use_hybrid=True)
    extractor = MetricsExtractor()
    metrics_repo = MetricsRepository(session_factory=None)

    # In extract() endpoint, before gather_extraction_chunks:
    window_start = req.filed_date - timedelta(days=30)
    window_end = req.filed_date + timedelta(days=30)
    chunks = await gather_extraction_chunks(retrieval, req.ticker, window_start, window_end)
    extracted = await extractor.extract(chunks, req.ticker, req.fiscal_period, req.filing_type, req.filed_date)
    # `gather_extraction_chunks` runs retrieval, which may invoke the
    # decomposer; that path does not surface a DecompositionResult here, so
    # only the extraction call is reported. Under-reporting by the
    # decomposer's share is the conservative direction and is noted rather
    # than silently accepted -- see trading-agent-known-gaps.md.
    _report_usage(response, extractor.last_usage)
    from app.infrastructure.repositories.metrics_repo import FinancialMetrics as MetricsRow
    row = MetricsRow(
        ticker=req.ticker,
        fiscal_period=req.fiscal_period,
        filing_type=req.filing_type,
        filed_date=req.filed_date,
        revenue=extracted.revenue,
        gross_margin_pct=extracted.gross_margin_pct,
        gaap_net_income=extracted.gaap_net_income,
        free_cash_flow=extracted.free_cash_flow,
        sbc_pct_of_revenue=extracted.sbc_pct_of_revenue,
        net_dollar_retention=extracted.net_dollar_retention,
        extraction_confidence=extracted.extraction_confidence,
        reasoning=extracted.reasoning,
        source_citations=[format_citation_tag(c) for c in chunks],
    )
    await metrics_repo.upsert(row)
    return extracted

async def gather_extraction_chunks(retrieval: RetrievalService, ticker: str, filed_after, filed_before):
    filters = ChunkSearchFilters(tickers=[ticker], filed_after=filed_after, filed_before=filed_before)
    seen: dict[int, RetrievedChunk] = {}
    for query in RetrievalService.METRIC_QUERIES.values():
        results = await retrieval.retrieve_hybrid(query, k=5, filters=filters)
        for chunk in results:
            cid = chunk.chunk.id
            if cid not in seen or chunk.similarity > seen[cid].similarity:
                seen[cid] = chunk
    return list(seen.values())

@app.get("/corpus-status")
async def corpus_status_endpoint(ticker: str | None = None):
    query = CorpusStatusQuery()
    summary = await query.summary(ticker)
    if not summary:
        return {"summary": [], "issues": [], "per_filing": []}

    issues = await query.issues(ticker)
    per_filing = await query.per_filing(ticker)
    
    return {
        "summary": [asdict(row) for row in summary],
        "issues": [asdict(i) for i in issues],
        "per_filing": [asdict(d) for d in per_filing],
    }

@app.post("/ingest")
async def ingest_endpoint(req: IngestRequest):
    user_agent = os.environ["EDGAR_USER_AGENT"]
    cache_root = Path(os.environ.get("EDGAR_CACHE_DIR", "./data/edgar-cache"))

    async with EdgarClient(user_agent, cache_root / "filings") as edgar:
        resolver = TickerResolver(user_agent, cache_root / "company_tickers.json")
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
        since = date(req.since_year, 1, 1) if req.since_year else None
        await service.ingest_security(
            ticker=req.ticker,
            form_types=[req.form_type] if req.form_type else None,
            limit=req.limit,
            since=since,
        )

    return {"status": "ok", "ticker": req.ticker, "limit": req.limit}


@app.post("/latest-filings")
async def latest_filings_endpoint(req: LatestFilingsRequest):
    user_agent = os.environ["EDGAR_USER_AGENT"]
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
        "total_on_sec": len(filings_list),
        "already_ingested": len(filings_list) - len(new_filings),
        "new_filings_count": len(new_filings),
        "filings": filings_list,
    }


@app.post("/news-assess", response_model=NewsAssessResponse)
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
    ticker: str
    thread_id: str | None = None

class TradingAnalysisResponse(BaseModel):
    ticker: str
    thread_id: str
    status: Literal["completed", "resumed", "started"]
    decision_memo: DecisionMemo


@app.post("/trading/analyze", response_model=TradingAnalysisResponse)
async def trading_analyze(req: TradingAnalysisRequest) -> TradingAnalysisResponse:
    if not req.ticker.strip():
        raise HTTPException(400, "ticker must not be empty")

    ticker = req.ticker.strip().upper()
    thread_id = req.thread_id or f"trading-{ticker}"
    graph = app.state.trading_graph
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)

    if state.values and not state.next:
        status: Literal["completed", "resumed", "started"] = "completed"
        result = state.values
    elif state.next:
        status = "resumed"
        result = await graph.ainvoke(None, config=config)
    else:
        status = "started"
        result = await graph.ainvoke({"ticker": ticker}, config=config)

    return TradingAnalysisResponse(
        ticker=ticker,
        thread_id=thread_id,
        status=status,
        decision_memo=result["decision_memo"],
    )