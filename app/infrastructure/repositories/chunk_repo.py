from app.domain.chunk import Chunk
from .db import get_connection

from dataclasses import dataclass
from datetime import date

@dataclass(frozen=True)
class ChunkSearchFilters:
    """Optional metadata filters applied before vector similarity ranking."""
    tickers: list[str] | None = None
    filing_types: list[str] | None = None
    filed_after: date | None = None
    filed_before: date | None = None
    section_path_contains: list[str] | None = None  # ANY of these in section_path


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from search, with similarity score."""
    chunk: Chunk
    similarity: float

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

    async def search_by_embedding(
        self,
        query_embedding: list[float],
        k: int = 10,
        filters: ChunkSearchFilters | None = None,
    ) -> list[RetrievedChunk]:
        """
        Vector similarity search with optional pre-filtering on metadata.

        Filters use indexed columns (ticker, filing_type, filed_date,
        section_path), so they execute as a Bitmap Index Scan BEFORE the
        HNSW similarity scan — that's why we denormalized those columns
        onto chunks.
        """
        filters = filters or ChunkSearchFilters()

        where_clauses: list[str] = ["embedding IS NOT NULL"]
        filter_params: list = []

        if filters.tickers:
            where_clauses.append("ticker = ANY(%s)")
            filter_params.append([t.upper() for t in filters.tickers])
        if filters.filing_types:
            where_clauses.append("filing_type = ANY(%s)")
            filter_params.append(filters.filing_types)
        if filters.filed_after:
            where_clauses.append("filed_date >= %s")
            filter_params.append(filters.filed_after)
        if filters.filed_before:
            where_clauses.append("filed_date <= %s")
            filter_params.append(filters.filed_before)
        if filters.section_path_contains:
            # ANY of the provided strings appears anywhere in section_path
            where_clauses.append("section_path && %s")   # array overlap operator
            filter_params.append(filters.section_path_contains)

        where_sql = " AND ".join(where_clauses)
        # Params in the same order as placeholders appear in the query:
        # 1) SELECT's similarity calc embedding
        # 2) WHERE filters (in order added)
        # 3) ORDER BY embedding
        # 4) LIMIT
        params = [query_embedding, *filter_params, query_embedding, k]

        query = f"""
            SELECT
                id, section_id, content, chunk_index, token_count, embedding,
                ticker, filed_date, filing_type, section_path, created_at,
                1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            WHERE {where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                rows = await cur.fetchall()

        results: list[RetrievedChunk] = []
        for row in rows:
            similarity = row.pop("similarity")
            chunk = Chunk.model_validate(row)
            results.append(RetrievedChunk(chunk=chunk, similarity=similarity))
        return results