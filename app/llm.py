import os 
from dotenv import load_dotenv
from anthropic import Anthropic
from app.chunk import Chunk, Chunks

load_dotenv(override=True)

claude_api_key = os.getenv("ANTHROPIC_API_KEY")
client  = Anthropic(api_key=claude_api_key)
claude_model = os.getenv("LLM_CLAUDE_MODEL")

SYSTEM_PROMPT = """You answer questions about SEC filings using ONLY the provided context chunks.
Rules:
- Every factual claim must cite the chunks(s) that support it, like [chunk 3] or [chunk 1, 4]
- If the context does not contain the answer, say so explicitly. Do not invent.
- Do not compute new financial figures. Quote numbers exactly as they appear
- Be concise, No premable. """

def answer(question: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(
        f"[chunk {c.chunk_index}] (similarity={c.similarity:.3f})\n{c.content}"
        for c in chunks
    )

    user_message = f"Context:\n\n{context}\n\nQuestion: {question}"

    resp = client.messages.create(
        model=claude_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{ "role": "user", "content": user_message }]
    )
    return resp.content[0].text
