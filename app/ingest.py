import os
import tiktoken
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI
from app.db_postgres import get_conn

load_dotenv(override=True)

open_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=open_api_key)

embedding_model = os.getenv("EMBEDDING_MODEL")
encoder = tiktoken.get_encoding("cl100k_base")

def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text:str, target_tokens: int = 600, overlap_tokens: int = 80) -> list[str]:
    tokens = encoder.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + target_tokens, len(tokens))
        chunks.append(encoder.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap_tokens
    return chunks

def embed_batch(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        input=texts,
        model=embedding_model
    )
    return [d.embedding for d in response.data]

def ingest_pdf(pdf_path: str, source_label: str) -> int:
    text = extract_text(pdf_path)
    chunks = chunk_text(text)

    embeddings = []
    for i in range(0, len(chunks), 100):
        embeddings.extend(embed_batch(chunks[i:i+100]))

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE source = %s", (source_label,))
        for idx, (content, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                "INSERT INTO chunks (source, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
                (source_label, idx, content, emb),
            )
        conn.commit()
    return len(chunks)