from domain.document import Document
from .db import get_connection

class DocumentRepository:
    async def insert(self, document: Document) -> Document:
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO documents (
                        filing_id, primary_document_name, document_type,
                        original_url, local_path
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id, created_at
                    """,
                    (
                        document.filing_id,
                        document.primary_document_name,
                        document.document_type,
                        document.original_url,
                        document.local_path,
                    ),
                )
                row = await cur.fetchone()
                await conn.commit()
        return document.model_copy(update={
            "id": row["id"],
            "created_at": row["created_at"],
        })

    async def list_for_filing(self, filing_id: int) -> list[Document]:
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT * FROM documents WHERE filing_id = %s ORDER BY id",
                    (filing_id,),
                )
                rows = await cur.fetchall()
        return [Document.model_validate(r) for r in rows]