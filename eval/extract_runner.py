from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from app.application.embedding_service import EmbeddingService
from app.application.extraction_service import MetricsExtractor
from app.application.query_decomposer import QueryDecomposer
from app.application.retrieval_service import RetrievalService
from app.infrastructure.repositories.chunk_repo import ChunkRepository, ChunkSearchFilters
from eval.question_embedding_cache import QuestionEmbeddingCache


CACHE_PATH = Path("eval/.extraction_embeddings_cache.json")

TOLERANCES = {
    "revenue": 1.0,
    "gross_margin_pct": 1.0,
    "free_cash_flow": 1.0,
    "sbc_pct_of_revenue": 1.0,
    "net_dollar_retention": 1.0,
}

@dataclass
class FieldResult:
    field: str
    expected: float
    actual: float | None
    passed: bool
    delta: float | None                         # None when actual is None
    confidence: str

@dataclass
class ExtractionCaseResult:
    ticker: str
    fiscal_period: str
    filed_date: date
    field_results: list[FieldResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.field_results)


async def _retrieve_for_extraction_cached(
    retrieval: RetrievalService,
    embedder: EmbeddingService,
    cache: QuestionEmbeddingCache,
    ticker: str,
    filed_date: date,
    k: int,
):
    """
    Same as RetrievalService.retrieve_for_extraction, but goes through the
    embedding cache — METRIC_QUERIES are fixed strings, so every case in
    every run re-embeds the same 4 queries unless cached.
    """
    filters = ChunkSearchFilters(
        tickers=[ticker],
        filing_types=None,
        filed_after=filed_date,
        filed_before=filed_date,
        section_path_contains=None,
    )
    collected = []
    for query in RetrievalService.METRIC_QUERIES.values():
        vector = await cache.get_or_embed(embedder, query)
        collected += await retrieval.retrieve_hybrid(vector, k=k, filters=filters)
    return retrieval._dedupe_by_chunk_id(collected)


async def run_extraction_eval(test_set_path: Path, k: int) -> list[ExtractionCaseResult]:
    embedder = EmbeddingService()
    retrieval = RetrievalService(
        embedding_service=embedder,
        chunk_repo=ChunkRepository(),
        decomposer=QueryDecomposer(),
    )
    extractor = MetricsExtractor()
    cache = QuestionEmbeddingCache(CACHE_PATH)

    cases = yaml.safe_load(test_set_path.read_text())
    results: list[ExtractionCaseResult] = []

    try:
        for case in cases:
            ticker = case["ticker"]
            period = case["fiscal_period"]
            filed_date = date.fromisoformat(case["filed_date"])

            chunks = await _retrieve_for_extraction_cached(
                retrieval, embedder, cache, ticker, filed_date, k
            )
            metrics = await extractor.extract(
                chunks, ticker, period, case["filing_type"], filed_date
            )

            field_results: list[FieldResult] = []
            for field, expected in case["expected"].items():
                actual = getattr(metrics, field)
                tolerance = TOLERANCES.get(field, 1.0)

                if actual is None:
                    field_results.append(FieldResult(
                        field=field, expected=expected, actual=None,
                        passed=False, delta=None,
                        confidence=metrics.extraction_confidence,
                    ))
                else:
                    delta = abs(actual - expected)
                    field_results.append(FieldResult(
                        field=field, expected=expected, actual=actual,
                        passed=delta <= tolerance, delta=delta,
                        confidence=metrics.extraction_confidence,
                    ))

            results.append(ExtractionCaseResult(
                ticker=ticker, fiscal_period=period,
                filed_date=filed_date, field_results=field_results,
            ))
    finally:
        cache.flush()

    return results