import asyncio
from datetime import date
import logging

from app.application.query_decomposer import DecompositionResult, QueryDecomposer
from app.infrastructure.repositories.chunk_repo import (
    ChunkRepository,
    ChunkSearchFilters,
    RetrievedChunk,
)
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


def _fuse_across_queries(
    per_query: list[list[RetrievedChunk]], k: int
) -> list[RetrievedChunk]:
    """Fuse several sub-queries' result lists into one ranking.

    Scores are SUMMED across sub-queries, not maxed. Each list is already
    RRF-fused within its own query, and an RRF score is rank-derived: taking
    the max across queries means "the best rank this chunk reached in any
    sub-query", which scores a chunk every sub-query ranked third exactly
    the same as one that a single sub-query ranked third and the rest missed
    entirely. Agreement across sub-queries is the whole reason to decompose
    a question, and the max threw it away.

    Summing is RRF's own rule applied one level up: a chunk answering two
    halves of a two-part question outranks one answering only a half.
    """
    scores: dict[int, float] = {}
    best: dict[int, RetrievedChunk] = {}
    for results in per_query:
        for chunk in results:
            cid = chunk.chunk.id
            scores[cid] = scores.get(cid, 0.0) + chunk.similarity
            # Keep the instance with the strongest single-query showing, so
            # `vector_similarity` is the best one this chunk actually got.
            if cid not in best or chunk.similarity > best[cid].similarity:
                best[cid] = chunk
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [
        RetrievedChunk(
            chunk=best[cid].chunk,
            similarity=score,
            vector_similarity=best[cid].vector_similarity,
        )
        for cid, score in ranked
    ]


class RetrievalService:
    """
    Orchestrates question → embedding → search → ranked chunks.

    Lives at the application layer because it composes multiple infrastructure
    services. Doesn't know about the LLM or how chunks become answers — that's
    the next layer up.
    """
    METRIC_QUERIES = {
        "income_statement": "revenue gross profit cost of revenue gross margin net income loss",
        "cash_flow": "free cash flow operating cash flow capital expenditures",
        "sbc": "stock-based compensation expense percentage of revenue",
        "retention": "net dollar retention rate customer retention",
    }
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        chunk_repo: ChunkRepository,
        decomposer: QueryDecomposer | None = None,
    ):
        # `use_hybrid` used to be a fourth argument. It was stored and never
        # read by anything in this class — callers pick hybrid by CALLING
        # retrieve_hybrid or retrieve_full — so four call sites were setting
        # a switch wired to nothing. That is the same trap models.py's
        # docstring describes for the two model env vars no code read.
        self.embedder = embedding_service
        self.chunk_repo = chunk_repo
        self.decomposer = decomposer

    async def retrieve(
        self,
        question: str,
        k: int = 8,
        filters: ChunkSearchFilters | None = None,
    ) -> list[RetrievedChunk]:
        if not question.strip():
            return []

        # Embed the question (single call, batch of 1)
        vectors = await self.embedder.embed_many([question])
        query_vector = vectors[0]
        return await self.retrieve_by_embedding(query_vector, k, filters)


    async def retrieve_by_embedding(
        self,
        query_embedding: list[float],
        k: int = 8,
        filters: ChunkSearchFilters | None = None,
    ) -> list[RetrievedChunk]:
        results = await self.chunk_repo.search_by_embedding(
            query_embedding=query_embedding,
            k=k,
            filters=filters,
        )
        if results:
            top = results[0]
            logger.info(
                "Retrieved %d chunks; top: %s @ %.3f",
                len(results),
                " > ".join(top.chunk.section_path),
                top.similarity,
            )
        else:
            logger.info("Retrieved 0 chunks (filters may be too narrow)")

        return results

    async def retrieve_hybrid(
        self,
        question: str,
        k: int = 8,
        filters: ChunkSearchFilters | None = None,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        """
        Hybrid retrieval: vector search + BM25, merged via
        reciprocal rank fusion.

        RRF score = 1/(rrf_k + rank_vector) + 1/(rrf_k + rank_bm25)

        rrf_k=60 is the standard constant from the original RRF paper
        (Cormack et al. 2009). Higher values dampen rank differences;
        lower values amplify them. 60 works well empirically for
        combining two diverse rankers.
        """
        # Retrieve more candidates from each path than we need,
        # so fusion has enough to work with
        candidate_k = k * 3

        # Vector path
        vectors = await self.embedder.embed_many([question])
        vector_results = await self.chunk_repo.search_by_embedding(
            query_embedding=vectors[0], k=candidate_k, filters=filters,
        )

        # BM25 path
        bm25_results = await self.chunk_repo.search_by_text(
            query=question, k=candidate_k, filters=filters,
        )

        # Reciprocal rank fusion
        merged = self._reciprocal_rank_fusion(
            vector_results, bm25_results, rrf_k=rrf_k, k=k,
        )

        logger.info(
            "Hybrid retrieval: %d vector + %d bm25 -> %d merged (top %d)",
            len(vector_results), len(bm25_results), len(merged), k,
        )

        return merged

    @staticmethod
    def _reciprocal_rank_fusion(
        vector_results: list[RetrievedChunk],
        bm25_results: list[RetrievedChunk],
        rrf_k: int = 60,
        k: int = 10,
    ) -> list[RetrievedChunk]:
        """
        Merge two ranked lists using reciprocal rank fusion.
        Chunks appearing in both lists get a combined score.
        Chunks appearing in only one list still participate.
        """
        scores: dict[int, float] = {}
        chunk_map: dict[int, RetrievedChunk] = {}

        for rank, chunk in enumerate(vector_results, start=1):
            cid = chunk.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunk_map[cid] = chunk

        for rank, chunk in enumerate(bm25_results, start=1):
            cid = chunk.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            if cid not in chunk_map:
                chunk_map[cid] = chunk

        # Sort by fused score descending, take top-k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        return [
            RetrievedChunk(
                chunk=chunk_map[cid].chunk,
                similarity=score,  # RRF score, not cosine similarity
                # Carried through from the vector side (None if only BM25
                # found it) — see RetrievedChunk.
                vector_similarity=chunk_map[cid].vector_similarity,
            )
            for cid, score in ranked
        ]
    
    async def retrieve_for_extraction(
        self,
        ticker: str,
        filed_date: date,
        k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Fixed-query retrieval for metric extraction.
        Scoped to a single filing via filed_date to prevent period bleeding.
        """
        filters = ChunkSearchFilters(
            tickers=[ticker],
            filing_types=None,
            filed_after=filed_date,
            filed_before=filed_date,
            section_path_contains=None,
        )
        # Four fixed queries, independent of each other. They ran one at a
        # time — the same serial pattern retrieve_full stopped using.
        per_query = await asyncio.gather(
            *(self.retrieve(query, k=k, filters=filters)
              for query in self.METRIC_QUERIES.values())
        )
        return self._dedupe_by_chunk_id([c for results in per_query for c in results])

    @staticmethod
    def _dedupe_by_chunk_id(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        best: dict[int, RetrievedChunk] = {}
        for c in chunks:
            key = c.chunk.id
            if key not in best or c.similarity > best[key].similarity:
                best[key] = c
        return list(best.values())

    async def retrieve_full(
        self,
        question: str,
        k: int = 8,
        filters: ChunkSearchFilters | None = None,
    ) -> tuple[list[RetrievedChunk], DecompositionResult]:
        """
        Full retrieval pipeline: decompose if needed, then hybrid-retrieve
        each sub-query, merge all results.
        """
        if self.decomposer is None:
            chunks = await self.retrieve_hybrid(question, k=k, filters=filters)
            result = DecompositionResult(
                original_query=question, was_decomposed=False, sub_queries=[question]
            )
            return chunks, result
    
        decomposition = await self.decomposer.decompose(question)
    
        if not decomposition.was_decomposed:
            chunks = await self.retrieve_hybrid(question, k=k, filters=filters)
            return chunks, decomposition
    
        # Hybrid-retrieve per sub-query, then fuse across them.
        # Independent of each other, so concurrent: each sub-query is an
        # embedding call plus two SQL searches, and a decomposed question has
        # 2-4 of them. One after another they added up on every /ask the
        # agent made.
        per_query = await asyncio.gather(
            *(self.retrieve_hybrid(sub_q, k=k, filters=filters) for sub_q in decomposition.sub_queries)
        )
        return _fuse_across_queries(per_query, k=k), decomposition