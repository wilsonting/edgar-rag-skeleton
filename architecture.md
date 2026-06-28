# Architecture: edgar-rag-skeleton

## Overview

`edgar-rag-skeleton` is a Retrieval-Augmented Generation (RAG) system for SEC EDGAR filings (10-K, 10-Q, 8-K, etc.). It ingests filings, parses and chunks their HTML into sections, embeds the chunks, and serves a question-answering API backed by citation-grounded LLM responses.

The codebase follows a clean architecture / DDD-style layering:

- **Domain** — framework-agnostic entities and value objects
- **Application** — orchestration services using domain + infrastructure
- **Infrastructure** — concrete implementations (DB, HTTP clients, parsing, chunking)

## Tech Stack

| Concern | Choice |
|---|---|
| API server | FastAPI + Uvicorn |
| LLM (answer generation) | Anthropic Claude  |
| Embeddings | OpenAI |
| Database | PostgreSQL + pgvector (HNSW index, cosine distance) |
| ORM/driver | SQLAlchemy 2.0 + asyncpg / psycopg async pool |
| Migrations | Alembic |
| HTML parsing | BeautifulSoup4 + lxml |
| Tokenization | tiktoken (cl100k_base) |
| CLI | Typer |


## Domain Model

**Filing state machine** (`domain/values.py`, `domain/filing.py`):

```
DISCOVERED -> DOWNLOADED -> PARSED -> CHUNKED -> EMBEDDED
                   \------------------------------> FAILED (resettable to DISCOVERED)
```

This makes ingestion resumable — each phase only processes filings sitting at the prerequisite status.

**Core entities:**
- `ListedSecurity` — a company (CIK, ticker, exchange, name)
- `Filing` — one filing (type, filed_date, accession_number, status)
- `Document` — pointer to a filing's downloaded HTML (local_path, original_url)
- `Section` — a structural division of a parsed filing (section_path array, content, order)
- `Chunk` — an embeddable unit with denormalized metadata (ticker, filed_date, filing_type, section_path) for fast pre-filtered vector search

## Database Schema

PostgreSQL 16 + pgvector. Five tables mirror the domain entities 1:1, with `chunks` denormalizing parent metadata for retrieval performance:

```
listed_securities (cik UNIQUE, ticker UNIQUE, exchange, name)
        │ 1:N
filings (security_id FK, filing_type, filed_date, accession_number UNIQUE, status enum, error_message)
        │ 1:N
documents (filing_id FK, primary_document_name, original_url, local_path)
        │ 1:N
sections (document_id FK, section_path TEXT[], order, content)
        │ 1:N
chunks (section_id FK, content, chunk_index, token_count, embedding vector(1536),
        ticker, filed_date, filing_type, section_path TEXT[])  -- denormalized for pre-filtering
```

Indexes:
- B-tree on `ticker`, `filed_date`, `filing_type` on `chunks` (cheap pre-filter before vector scan)
- GIN on `section_path` (array containment, e.g. filter by "Risk Factors")
- HNSW on `embedding` with `vector_cosine_ops` (approximate nearest-neighbor search)

Vector search query shape: apply metadata `WHERE` filters first (bitmap index scan), then `ORDER BY embedding <=> query_vector LIMIT k`, with similarity computed as `1 - cosine_distance`.

## Ingestion Pipeline

Orchestrated by `IngestionService` (`application/ingestion_service.py`), driven by the CLI (`ingest` command) per ticker:

1. **Discover** — `TickerResolver` resolves ticker → CIK; `EdgarClient.list_filings()` queries SEC submissions API; filings upserted with status `DISCOVERED`.
2. **Download** — `EdgarClient.download_filing()` fetches and caches HTML under `data/edgar-cache/`; `Document` row created; status → `DOWNLOADED`.
3. **Parse** — `filing_parser.parse_filing()` strips noise (scripts/styles/XBRL), flattens DOM to text blocks, locates `Item N` headings via regex, dedupes table-of-contents repeats (keeps last occurrence), slices into `ParsedSection`s; status → `PARSED`.
4. **Chunk** — `section_chunker.chunk_filing()` splits each section into ~600-token chunks on paragraph boundaries with 80-token overlap between adjacent chunks, never crossing section boundaries, dropping trivial (<50 token) sections; status → `CHUNKED`.
5. **Embed** — `EmbeddingService` batches chunk content through OpenAI embeddings; vectors written back via `ChunkRepository.update_embeddings()`; status → `EMBEDDED`.

SEC EDGAR access is rate-limited client-side (~8 req/sec) and requires a configured `User-Agent`.

## Query Pipeline (`POST /ask`)

Handled in `main.py`, using `RetrievalService` and `llm.answer_question()`:

1. Validate `AskRequest` (question, k, optional filters: tickers, filing_types, filed_after/before, section_path_contains).
2. Embed the question (`EmbeddingService`).
3. `ChunkRepository.search_by_embedding()` — filtered HNSW vector search returns top-k `RetrievedChunk`s with similarity scores.
4. `citations.format_context_block()` builds a context string tagging each chunk as `[TICKER FORM YEAR §Item]`.
5. `llm.answer_question()` calls Claude with a system prompt that requires citing every fact and forbids speculation; returns answer text.
6. Citation tags actually present in the answer are extracted by substring match against expected tags.
7. Response (`AskResponse`) bundles the answer, cited tags, and the source chunks (with previews and similarity scores).

## External Services

| Service | Used for | Auth |
|---|---|---|
| SEC EDGAR (`data.sec.gov`, `www.sec.gov`) | Filing discovery + HTML download | `User-Agent` header (`EDGAR_USER_AGENT`) |
| OpenAI | Embeddings (`text-embedding-3-small`) | `OPENAI_API_KEY` |
| Anthropic | Answer generation (Claude) | `ANTHROPIC_API_KEY` |

## Design Notes

- **Resumable ingestion** via the `Filing` status state machine — reruns pick up at the last completed phase instead of redoing work.
- **Denormalized chunk metadata** trades storage for retrieval latency: metadata filters apply as cheap indexed predicates before the more expensive HNSW vector scan.
- **TOC deduplication** in the parser (`filing_parser._locate_item_headings`) is a load-bearing detail — without it, sections capture empty table-of-contents entries instead of actual content, since EDGAR filings repeat "Item N" headings in both the TOC and the body.
- **Citation format is intentionally lightweight** (`[TICKER FORM YEAR §Item]`, no chunk ID) — sufficient for traceability without complicating the prompt.
- **Aggregate lifecycle vs read-side queries** - Repositories handle aggregate lifecycle (insert, update, find). Cross-aggregate read queries live under infrastructure/queries/ to keep write-side and read-side concerns visibly separate.
- Read-side queries return frozen dataclasses rather than dicts — moves field-name errors from runtime puzzles to static-checker / immediate-crash failures.