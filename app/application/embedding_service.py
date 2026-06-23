import os
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Wraps OpenAI's embeddings API with batching."""

    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 100):
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
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