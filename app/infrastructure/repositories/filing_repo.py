from datetime import date

from app.domain.filing import Filing
from app.domain.values import FilingStatus
from .db import get_connection

class FilingRepository:
    async def upsert(self, filing: Filing) -> Filing:
        """Insert or update by accession_number (natural key)."""
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO filings (
                        security_id, filing_type, filed_date, period_of_report,
                        accession_number, status, error_message
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (accession_number) DO UPDATE
                        SET status = EXCLUDED.status,
                            error_message = EXCLUDED.error_message,
                            updated_at = now()
                    RETURNING id, created_at, updated_at
                    """,
                    (
                        filing.security_id,
                        filing.filing_type,
                        filing.filed_date,
                        filing.period_of_report,
                        filing.accession_number,
                        filing.status.value,
                        filing.error_message,
                    ),
                )
                row = await cur.fetchone()
                await conn.commit()
        return filing.model_copy(update={
            "id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        })

    async def mark_status(
        self,
        filing_id: int,
        status: FilingStatus,
        error_message: str | None = None,
    ) -> None:
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE filings
                    SET status = %s, error_message = %s, updated_at = now()
                    WHERE id = %s
                    """,
                    (status.value, error_message, filing_id),
                )
                await conn.commit()

    async def list_by_status(
        self,
        statuses: list[FilingStatus],
        limit: int | None = None,
    ) -> list[Filing]:
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                query = """
                    SELECT * FROM filings
                    WHERE status = ANY(%s)
                    ORDER BY filed_date DESC
                """
                params: tuple = ([s.value for s in statuses],)
                if limit:
                    query += " LIMIT %s"
                    params = (*params, limit)
                await cur.execute(query, params)
                rows = await cur.fetchall()
        return [Filing.model_validate(r) for r in rows]

    async def get_by_accession(self, accession_number: str) -> Filing | None:
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT * FROM filings WHERE accession_number = %s",
                    (accession_number,),
                )
                row = await cur.fetchone()
        return Filing.model_validate(row) if row else None