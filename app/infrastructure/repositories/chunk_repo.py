from app.domain.chunk import Chunk
from .db import get_connection


class ChunkRepository:
    async def bulk_insert(self, chunks: list[Chunk]) -> list[Chunk]:
        """Insert chunks (typically without embeddings yet)."""
        if not chunks:
            return []

        async with get_connection() as conn:
            async with conn.cursor() as cur:
                values = [
                    (
                        c.section_id, c.content, c.chunk_index, c.token_count,
                        c.embedding,
                        c.ticker, c.filed_date, c.filing_type, c.section_path,
                    )
                    for c in chunks
                ]
                await cur.executemany(
                    """
                    INSERT INTO chunks (
                        section_id, content, chunk_index, token_count, embedding,
                        ticker, filed_date, filing_type, section_path
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, created_at
                    """,
                    values,
                    returning=True,
                )

                results: list[Chunk] = []
                idx = 0
                while True:
                    row = await cur.fetchone()
                    if row is not None:
                        results.append(chunks[idx].model_copy(update={
                            "id": row["id"],
                            "created_at": row["created_at"],
                        }))
                        idx += 1
                    if not cur.nextset():
                        break
                await conn.commit()

        return results

    async def list_without_embeddings(
        self,
        filing_id: int | None = None,
        limit: int = 1000,
    ) -> list[Chunk]:
        """Resumability: chunks ready to be embedded."""
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                query = """
                    SELECT c.* FROM chunks c
                    JOIN sections s ON s.id = c.section_id
                    JOIN documents d ON d.id = s.document_id
                    WHERE c.embedding IS NULL
                """
                params: tuple = ()
                if filing_id is not None:
                    query += " AND d.filing_id = %s"
                    params = (filing_id,)
                query += " ORDER BY c.id LIMIT %s"
                params = (*params, limit)

                await cur.execute(query, params)
                rows = await cur.fetchall()
        return [Chunk.model_validate(r) for r in rows]

    async def update_embeddings(self, updates: list[tuple[int, list[float]]]) -> None:
        """Set embeddings on existing chunks. updates: list of (chunk_id, vector)."""
        if not updates:
            return
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    "UPDATE chunks SET embedding = %s WHERE id = %s",
                    [(emb, cid) for cid, emb in updates],
                )
                await conn.commit()