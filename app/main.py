from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import date

from app.application.embedding_service import EmbeddingService
from app.application.retrieval_service import RetrievalService
from app.infrastructure.repositories.db import init_pool, close_pool
from app.infrastructure.repositories.chunk_repo import (
    ChunkRepository,
    ChunkSearchFilters,
)
from app.llm import answer_question


app = FastAPI(title="RAG Skeleton")

@app.on_event("startup")
def startup():
    init_pool()

@app.on_event("shutdown")
async def _shutdown() -> None:
    await close_pool()

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
    chunks: list[RetrievedChunkResponse]

# ---- Endpoint ----
@app.post("/ask",  response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")

    embedder = EmbeddingService()
    chunk_repo = ChunkRepository()
    retrieval = RetrievalService(embedder, chunk_repo)

    filters = ChunkSearchFilters(
        tickers=req.tickers,
        filing_types=req.filing_types,
        filed_after=req.filed_after,
        filed_before=req.filed_before,
        section_path_contains=req.section_path_contains,
    )

    chunks = await retrieval.retrieve(req.question, k=req.k, filters=filters)
    result = await answer_question(req.question, chunks)

    from app.application.citations import format_citation_tag
    return AskResponse(
        answer=result.answer,
        citations=result.citations,
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
