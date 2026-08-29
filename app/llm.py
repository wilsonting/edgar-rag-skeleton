import logging
from dotenv import load_dotenv

from app.domain.token_usage import TokenUsage
from dataclasses import dataclass, field
from app.chunk import Chunk, Chunks
from app.infrastructure.repositories.chunk_repo import RetrievedChunk
from app.application.citations import format_citation_tag, format_context_block
from app.infrastructure.llm import get_client
from app.infrastructure.llm.models import model_for

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# The default for callers that do not pass one. `answer_question` takes
# `model` as a parameter, so this is only the fallback.
claude_model = model_for("answer")

SYSTEM_PROMPT = """You answer questions about SEC filings using ONLY the provided context excerpts.
Rules:
- Every factual claim must cite the source excerpt(s) using the tags in the context, like [AAPL 10-K 2025 §Item 1A]. Multiple sources: [AAPL 10-K 2025 §Item 1A, MSFT 10-K 2024 §Item 7].
- If the context does not contain the enough information, say so explicitly. 
  Do not speculate or fall back on generate knowledge.
- Quote numbers exactly as they appear, with the citation. You may compute simple ratios, margins, percentages, and growth rates from figures explicitly stated in the excerpts (e.g., operating margin from operating income ÷ revenue), but never estimate or fabricate figures not present in the context.
- Be concise, No premable. no "Based on the provided context,
- If multiple companies are involved, organize the answer by company
- When computing a ratio or growth rate, state both input values and the
  periods they come from. Never combine figures from different fiscal
  periods or different metrics in one calculation.
"""

@dataclass(frozen=True)
class AnswerWithCitations:
    answer: str
    citations: list[str]   # The tags that appeared in the answer
    chunks: list[RetrievedChunk]   # All chunks supplied to the LLM
    # What this call cost. Reported rather than logged here: llm.py has no
    # run_id and no state, and the caller that owns the run is the only
    # place the number can be attributed correctly. See domain/token_usage.
    usage: TokenUsage = field(default_factory=TokenUsage)

async def answer_question(
    question: str,
    chunks: list[RetrievedChunk],
    model: str,
    max_tokens: int = 1024,
) -> AnswerWithCitations:
    if not chunks:
        # No LLM call happens on this path, so usage is genuinely zero --
        # not unknown.
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

    # Resolved per call rather than once at import: `model` is what
    # decides the provider now, and it is a parameter here.
    resp = await get_client(model).messages.create(
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
        usage=TokenUsage.from_response(resp.usage),
    )
