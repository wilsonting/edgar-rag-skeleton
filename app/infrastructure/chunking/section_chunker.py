import tiktoken
from app.infrastructure.parsing.models import ParsedSection
from .models import ChunkDraft

# cl100k_base matches GPT-4 / text-embedding-3-* tokenization closely enough
# for chunk-size planning purposes.
_ENCODER = tiktoken.get_encoding("cl100k_base")


def chunk_filing(
    sections: list[ParsedSection],
    target_tokens: int = 600,
    overlap_tokens: int = 80,
    min_chunk_tokens: int = 50,
) -> list[ChunkDraft]:
    """
    Produce chunks from parsed sections. Rules:
      1. Chunks never cross section boundaries.
      2. Paragraphs are kept whole when they fit.
      3. Overlap is added between adjacent chunks of the same section.
      4. Very small sections become single small chunks.
    """
    drafts: list[ChunkDraft] = []
    global_index = 0

    for section in sections:
        for chunk_text, token_count in _chunk_one_section(
            section.content, target_tokens, overlap_tokens, min_chunk_tokens
        ):
            drafts.append(
                ChunkDraft(
                    section_path=section.section_path,
                    chunk_index=global_index,
                    content=chunk_text,
                    token_count=token_count,
                )
            )
            global_index += 1

    return drafts


def _chunk_one_section(
    text: str,
    target_tokens: int,
    overlap_tokens: int,
    min_chunk_tokens: int,
) -> list[tuple[str, int]]:
    """Split one section's text into (chunk_text, token_count) pairs."""
    tokens = _ENCODER.encode(text)
    if len(tokens) <= target_tokens:
        if len(tokens) < min_chunk_tokens:
            return []   # drop trivially small sections (headers only, etc.)
        return [(text, len(tokens))]

    # Prefer to split on paragraph boundaries; fall back to token windows.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return _token_window_split(tokens, target_tokens, overlap_tokens)

    chunks: list[tuple[str, int]] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for para in paragraphs:
        para_tokens = len(_ENCODER.encode(para))

        if para_tokens > target_tokens:
            # Flush buffer first
            if buffer:
                joined = "\n\n".join(buffer)
                chunks.append((joined, buffer_tokens))
                buffer, buffer_tokens = [], 0
            # Then split the oversized paragraph by token window
            chunks.extend(
                _token_window_split(
                    _ENCODER.encode(para), target_tokens, overlap_tokens
                )
            )
            continue

        if buffer_tokens + para_tokens > target_tokens:
            joined = "\n\n".join(buffer)
            chunks.append((joined, buffer_tokens))
            # Start new buffer with overlap from the tail of the previous chunk
            buffer, buffer_tokens = _carry_overlap(joined, overlap_tokens)
            buffer.append(para)
            buffer_tokens += para_tokens
        else:
            buffer.append(para)
            buffer_tokens += para_tokens

    if buffer and buffer_tokens >= min_chunk_tokens:
        chunks.append(("\n\n".join(buffer), buffer_tokens))

    return chunks


def _token_window_split(
    tokens: list[int], target_tokens: int, overlap_tokens: int
) -> list[tuple[str, int]]:
    """Sliding token window — last resort for huge unbroken paragraphs."""
    chunks: list[tuple[str, int]] = []
    start = 0
    while start < len(tokens):
        end = min(start + target_tokens, len(tokens))
        slice_ = tokens[start:end]
        chunks.append((_ENCODER.decode(slice_), len(slice_)))
        if end == len(tokens):
            break
        start = end - overlap_tokens
    return chunks


def _carry_overlap(prev_chunk_text: str, overlap_tokens: int) -> tuple[list[str], int]:
    """Take the tail of the previous chunk as overlap into the next buffer."""
    tokens = _ENCODER.encode(prev_chunk_text)
    if len(tokens) <= overlap_tokens:
        return [prev_chunk_text], len(tokens)
    tail = _ENCODER.decode(tokens[-overlap_tokens:])
    return [tail], overlap_tokens