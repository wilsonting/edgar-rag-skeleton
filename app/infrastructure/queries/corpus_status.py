from app.infrastructure.queries.models import CorpusSummaryRow, FilingDetail, FilingIssue
from app.infrastructure.repositories.db import get_connection


class CorpusStatusQuery:
    """Read-side query for corpus introspection. Not a repository."""

    async def summary(self, ticker: str | None = None) -> list[CorpusSummaryRow]:
        where_clause = "WHERE s.ticker = %s" if ticker else ""
        params: tuple = (ticker.upper(),) if ticker else ()

        async with get_connection() as conn, conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT
                    s.ticker,
                    count(DISTINCT f.id) AS filings,
                    min(f.filed_date) AS earliest,
                    max(f.filed_date) AS latest,
                    count(DISTINCT f.id) FILTER (WHERE f.status = 'embedded') AS embedded,
                    count(DISTINCT f.id) FILTER (
                        WHERE f.status NOT IN ('embedded', 'failed')
                    ) AS partial,
                    count(DISTINCT f.id) FILTER (WHERE f.status = 'failed') AS failed,
                    count(c.id) AS chunks,
                    count(c.id) FILTER (WHERE c.embedding IS NOT NULL) AS embedded_chunks
                FROM listed_securities s
                LEFT JOIN filings f ON f.security_id = s.id
                LEFT JOIN documents d ON d.filing_id = f.id
                LEFT JOIN sections sec ON sec.document_id = d.id
                LEFT JOIN chunks c ON c.section_id = sec.id
                {where_clause}
                GROUP BY s.ticker
                ORDER BY s.ticker
                """,
                params,
            )
            rows = await cur.fetchall()

        return [CorpusSummaryRow(**r) for r in rows]

    async def issues(self, ticker: str | None = None) -> list[FilingIssue]:
        where_clause = "WHERE s.ticker = %s AND" if ticker else "WHERE"
        params: tuple = (ticker.upper(),) if ticker else ()

        async with get_connection() as conn, conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT s.ticker, f.accession_number, f.filing_type,
                       f.filed_date, f.status::text AS status,
                       f.error_message, f.updated_at
                FROM filings f
                JOIN listed_securities s ON s.id = f.security_id
                {where_clause} (
                    f.status NOT IN ('embedded', 'failed')
                    OR f.status = 'failed'
                )
                ORDER BY f.updated_at
                """,
                params,
            )
            rows = await cur.fetchall()

        return [FilingIssue(**r) for r in rows]

    async def per_filing(self, ticker: str | None = None) -> list[FilingDetail]:
        where_clause = "WHERE s.ticker = %s" if ticker else ""
        params: tuple = (ticker.upper(),) if ticker else ()

        async with get_connection() as conn, conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT s.ticker, f.accession_number, f.filing_type,
                       f.filed_date, f.status::text AS status,
                       count(c.id) AS chunks,
                       count(c.id) FILTER (WHERE c.embedding IS NOT NULL) AS embedded
                FROM filings f
                JOIN listed_securities s ON s.id = f.security_id
                LEFT JOIN documents d ON d.filing_id = f.id
                LEFT JOIN sections sec ON sec.document_id = d.id
                LEFT JOIN chunks c ON c.section_id = sec.id
                {where_clause}
                GROUP BY s.ticker, f.accession_number, f.filing_type,
                         f.filed_date, f.status
                ORDER BY s.ticker, f.filed_date DESC
                """,
                params,
            )
            rows = await cur.fetchall()

        return [FilingDetail(**r) for r in rows]