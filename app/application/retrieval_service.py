import logging

from app.infrastructure.repositories.chunk_repo import (
    ChunkRepository,
    ChunkSearchFilters,
    RetrievedChunk,
)
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Orchestrates question → embedding → search → ranked chunks.

    Lives at the application layer because it composes multiple infrastructure
    services. Doesn't know about the LLM or how chunks become answers — that's
    the next layer up.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        chunk_repo: ChunkRepository,
    ):
        self.embedder = embedding_service
        self.chunk_repo = chunk_repo

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

        results = await self.chunk_repo.search_by_embedding(
            query_embedding=query_vector,
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