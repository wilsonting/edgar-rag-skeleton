import os
import psycopg
from app.chunk import Chunk
from app.db_postgres import get_conn
from openai import OpenAI

open_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=open_api_key)

embedding_model = os.getenv("EMBEDDING_MODEL")

def retrieve(query: str, k: int = 5) -> list[dict]:
    query_embedding = openai_client.embeddings.create(
        model=embedding_model, input=[query]
    ).data[0].embedding

    with get_conn() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            """
            SELECT id, source, chunk_index, content,
                1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s          
            """,
            (query_embedding, query_embedding, k),
        )
        rows = cur.fetchall()
    
    for r in rows:
        print(r)

    return [Chunk.model_validate(r) for r in rows]