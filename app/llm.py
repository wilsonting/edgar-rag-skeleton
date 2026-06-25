import os 
import logging
from dotenv import load_dotenv
from dataclasses import dataclass
from anthropic import AsyncAnthropic
from app.chunk import Chunk, Chunks
from app.infrastructure.repositories.chunk_repo import RetrievedChunk
from app.application.citations import format_citation_tag, format_context_block

load_dotenv(override=True)

logger = logging.getLogger(__name__)

claude_api_key = os.getenv("ANTHROPIC_API_KEY")
_client  = AsyncAnthropic(api_key=claude_api_key)
claude_model = os.getenv("LLM_CLAUDE_MODEL")

SYSTEM_PROMPT = """You answer questions about SEC filings using ONLY the provided context excerpts.
Rules:
- Every factual claim must cite the source excerpt(s) using the tags in the context, like [AAPL 10-K 2025 §Item 1A]. Multiple sources: [AAPL 10-K 2025 §Item 1A, MSFT 10-K 2024 §Item 7].
- If the context does not contain the enough information, say so explicitly. 
  Do not speculate or fall back on generate knowledge.
- Do not compute or estimate financial figures. Quote numbers exactly as they appear, with the citation
- Be concise, No premable. no "Based on the provided context,
- If multiple companies are involved, organize the answer by company
"""

@dataclass(frozen=True)
class AnswerWithCitations:
    answer: str
    citations: list[str]   # The tags that appeared in the answer
    chunks: list[RetrievedChunk]   # All chunks supplied to the LLM

async def answer_question(
    question: str,
    chunks: list[RetrievedChunk],
    model: str = "claude-opus-4-7",
    max_tokens: int = 1024,
) -> AnswerWithCitations:
    if not chunks:
        return AnswerWithCitations(
            answer="No relevant excerpts were found for this question.",
            citations=[],
            chunks=[],
        )

    context_block = format_context_block(chunks)
    user_message = (
        f"Context excerpts:\n\n{context_block}\n\n"
        f"Question: {question}"
    )

    resp = await _client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    answer_text = resp.content[0].text

    # Lightweight citation extraction — find tags actually mentioned in answer
    expected_tags = {format_citation_tag(c) for c in chunks}
    cited = [tag for tag in expected_tags if tag in answer_text]

    return AnswerWithCitations(
        answer=answer_text,
        citations=cited,
        chunks=chunks,
    )
