import os
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector


load_dotenv(override=True)
DATABASE_URL = os.environ["POSTGRES_DATABASE_URL"]

def get_conn():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn

def init_schema():
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
    with get_conn() as conn: 
        with conn.cursor() as cur:
            # pgvector's vector(1536) matches text-embedding-3-small's output dimension, 
            # and the HNSW index is what makes similarity search fast at scale.
            cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                source TEXT NOT NULL,
                chunk_index INT NOT NULL,
                content TEXT NOT NULL,
                embedding vector(1536) NOT NULL
            )
            """)
            cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks USING hnsw (embedding vector_cosine_ops)
            """)
        conn.commit()