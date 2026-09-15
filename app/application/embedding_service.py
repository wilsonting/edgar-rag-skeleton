import os
import logging
from openai import AsyncOpenAI

from app.config import require_env

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Wraps OpenAI's embeddings API with batching."""

    def __init__(self, model: str | None = None, batch_size: int = 100):
        self.client = AsyncOpenAI(api_key=require_env("OPENAI_API_KEY"))
        # EMBEDDING_MODEL was in .env.example, and read only by the dead PDF
        # pipeline (app/ingest.py, app/retrieve.py) — setting it looked like it
        # worked and did nothing. The chunks.embedding column is vector(1536),
        # so a replacement must produce 1536 dimensions, and changing it means
        # re-embedding the corpus: vectors from two models are not comparable.
        self.model = model or os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
        self.batch_size = batch_size

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns vectors in the same order."""
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            logger.debug("Embedding batch %d-%d of %d", i, i + len(batch), len(texts))
            resp = await self.client.embeddings.create(model=self.model, input=batch)
            all_vectors.extend(d.embedding for d in resp.data)
        return all_vectors