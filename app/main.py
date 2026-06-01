from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.ingest import ingest_pdf
from app.retrieve import retrieve
from app.llm import answer
from app.db_postgres import init_schema

app = FastAPI(title="RAG Skeleton")

@app.on_event("startup")
def startup():
    init_schema()

class IngestRequest(BaseModel):
    pdf_path: str
    source_label: str

class AskRequest(BaseModel):
    question: str
    k: int = 5

@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        count = ingest_pdf(req.pdf_path, req.source_label)
        return {"status": "ok", "chunks_ingested": count}
    except FileNotFoundError:
        raise HTTPException(404, f"PDF not found: {req.pdf_path}")

@app.post("/ask")
def ask(req: AskRequest):
    chunks = retrieve(req.question, k=req.k)
    if not chunks:
        return {"answer": "No documents have been ingested yet.", "chunks": []}
    response = answer(req.question, chunks)
    return {
        "answer": response,
        "chunks": [
            {"chunk_index": c.chunk_index, "source": c.source, "similarity": c.similarity}
            for c in chunks
        ]
    }
