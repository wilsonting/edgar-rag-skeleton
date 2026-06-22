from psycopg.types.json import Json
from domain.section import Section
from .db import get_connection


class SectionRepository:
    async def bulk_insert(self, sections: list[Section]) -> list[Section]:
        """Insert many sections at once. Returns them with ids populated."""
        if not sections:
            return []

        async with get_connection() as conn:
            async with conn.cursor() as cur:
                # executemany returning IDs requires RETURNING — Postgres
                # supports it via INSERT ... RETURNING per row.
                values = [
                    (s.document_id, s.section_path, s.order, s.content)
                    for s in sections
                ]
                await cur.executemany(
                    """
                    INSERT INTO sections (document_id, section_path, "order", content)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, created_at
                    """,
                    values,
                    returning=True,
                )

                results: list[Section] = []
                idx = 0
                while True:
                    row = await cur.fetchone()
                    if row is not None:
                        results.append(sections[idx].model_copy(update={
                            "id": row["id"],
                            "created_at": row["created_at"],
                        }))
                        idx += 1
                    if not cur.nextset():
                        break
                await conn.commit()

        return results

    async def list_for_document(self, document_id: int) -> list[Section]:
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    'SELECT * FROM sections WHERE document_id = %s ORDER BY "order"',
                    (document_id,),
                )
                rows = await cur.fetchall()
        return [Section.model_validate(r) for r in rows]