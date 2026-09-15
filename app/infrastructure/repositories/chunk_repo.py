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
    """A chunk returned from search, with its ranking score.

    `similarity` is whatever the search ranked by: cosine similarity for a
    vector search, term coverage for a keyword search, the fused RRF score
    for hybrid. `vector_similarity` is always the cosine similarity when the
    chunk came through the vector search, and None when only the keyword
    search found it — the one score comparable across queries, and the one
    a reader can interpret (an RRF score of 0.016 says nothing about fit).
    """
    chunk: Chunk
    similarity: float
    vector_similarity: float | None = None

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

    async def delete_for_sections(self, section_ids: list[int]) -> int:
        """Delete every chunk of these sections. Returns how many went.

        What makes re-chunking idempotent: IngestionService._chunk clears a
        document's chunks before inserting, so a run that died after the
        insert but before the filing was marked CHUNKED re-chunks cleanly
        instead of adding a second copy of every chunk.
        """
        if not section_ids:
            return 0
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM chunks WHERE section_id = ANY(%s)", (section_ids,)
                )
                deleted = cur.rowcount
                await conn.commit()
        return deleted

    async def sections_with_content(
        self, sections: list[str], tickers: list[str] | None = None
    ) -> set[str]:
        """Which of `sections` appear in some chunk's section path.

        A section filter naming something the corpus doesn't have would
        silently return nothing — and to the caller an empty answer is
        indistinguishable from "the filing doesn't say". Cheap enough to
        check before spending the retrieval: one index scan, no embedding.
        """
        if not sections:
            return set()
        # Named, because the pool's row factory is dict_row.
        sql = (
            "SELECT DISTINCT unnest(section_path) AS part "
            "FROM chunks WHERE section_path && %s"
        )
        params: list = [sections]
        if tickers:
            sql += " AND ticker = ANY(%s)"
            params.append([t.upper() for t in tickers])   # as search_by_embedding does
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                found = {row["part"] for row in await cur.fetchall()}
        return {s for s in sections if s in found}

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
            if row["embedding"] is not None:
                row["embedding"] = row["embedding"].to_list()
            chunk = Chunk.model_validate(row)
            results.append(RetrievedChunk(
                chunk=chunk, similarity=similarity, vector_similarity=similarity,
            ))
        return results

    async def search_by_text(
        self,
        query: str,
        k: int = 20,
        filters: ChunkSearchFilters | None = None,
    ) -> list[RetrievedChunk]:
        """
        Keyword search over Postgres tsvector: a chunk matches if it contains
        ANY query term, ranked by ts_rank.

        Was `plainto_tsquery` alone, which joins every term with AND: a chunk
        had to contain every word of the question. The metric queries are
        nine-word bags and the agent's questions are long compound
        sentences, so it matched nothing — 40 of 43 hybrid retrievals on the
        2026-09-11 ACN run, and 51 of 51 real queries replayed against the
        corpus, came back with zero hits. "Hybrid" retrieval was vector-only.

        The OR query is plainto_tsquery's own output with `&` rejoined as
        `|`, so parsing, stemming, stopwords and escaping are unchanged.
        Plain ts_rank was measured against two alternatives that should have
        done better on paper — ranking by distinct-term coverage, and
        IDF-weighted coverage (ts_rank has no IDF) — on the same 51 queries,
        counting top-8 chunks that actually contain the answer. After fusion:
        cash-flow questions 42 / 38 / 35 answering chunks (vector-only: 37),
        internal-control questions 11 / 12 / 11 (vector-only: 6). The
        simplest one is no worse, so it is the one here.
        """
        filters = filters or ChunkSearchFilters()

        where_clauses: list[str] = ["c.content_tsv @@ q.any_term"]
        params: list = [query]

        if filters.tickers:
            where_clauses.append("c.ticker = ANY(%s)")
            params.append([t.upper() for t in filters.tickers])
        if filters.filing_types:
            where_clauses.append("c.filing_type = ANY(%s)")
            params.append(filters.filing_types)
        if filters.filed_after:
            where_clauses.append("c.filed_date >= %s")
            params.append(filters.filed_after)
        if filters.filed_before:
            where_clauses.append("c.filed_date <= %s")
            params.append(filters.filed_before)
        if filters.section_path_contains:
            where_clauses.append("c.section_path && %s")
            params.append(filters.section_path_contains)

        where_sql = " AND ".join(where_clauses)
        params.append(k)

        sql = f"""
            WITH q AS (
                SELECT replace(plainto_tsquery('english', %s)::text, ' & ', ' | ')::tsquery
                    AS any_term
            )
            SELECT
                c.id, c.section_id, c.content, c.chunk_index, c.token_count, c.embedding,
                c.ticker, c.filed_date, c.filing_type, c.section_path, c.created_at,
                ts_rank(c.content_tsv, q.any_term) AS similarity
            FROM chunks c, q
            WHERE {where_sql}
            ORDER BY similarity DESC, c.id
            LIMIT %s
        """

        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
    
        results: list[RetrievedChunk] = []
        for row in rows:
            similarity = row.pop("similarity")
            if row["embedding"] is not None:
                row["embedding"] = row["embedding"].to_list()
            chunk = Chunk.model_validate(row)
            results.append(RetrievedChunk(chunk=chunk, similarity=float(similarity)))
        return results