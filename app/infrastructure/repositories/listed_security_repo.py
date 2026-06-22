from domain.listed_security import ListedSecurity
from .db import get_connection

class ListedSecurityRepository:
    """Persistence for ListedSecurity."""

    async def upsert(self, security: ListedSecurity) -> ListedSecurity:
        """
        Insert or update by CIK (the natural key). Returns the persisted entity
        with id populated.
        """
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO listed_securities (cik, ticker, exchange, name)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (cik) DO UPDATE
                        SET ticker = EXCLUDED.ticker,
                            exchange = EXCLUDED.exchange,
                            name = EXCLUDED.name,
                            updated_at = now()
                    RETURNING id, created_at, updated_at
                    """,
                    (security.cik, security.ticker, security.exchange, security.name),
                )
                row = await cur.fetchone()
                await conn.commit()

        return security.model_copy(update={
            "id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        })

    async def get_by_ticker(self, ticker: str) -> ListedSecurity | None:
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT * FROM listed_securities WHERE ticker = %s",
                    (ticker.upper(),),
                )
                row = await cur.fetchone()
        return ListedSecurity.model_validate(row) if row else None

    async def get_by_cik(self, cik: str) -> ListedSecurity | None:
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT * FROM listed_securities WHERE cik = %s",
                    (cik,),
                )
                row = await cur.fetchone()
        return ListedSecurity.model_validate(row) if row else None