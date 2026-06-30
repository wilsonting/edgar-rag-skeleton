# Architecture: edgar-rag-skeleton

## Overview

`edgar-rag-skeleton` is an end-to-end pipeline that ingests SEC EDGAR filings (10-K, 10-Q, 8-K, etc.), parses and chunks them with section-aware structure, embeds them into pgvector, and serves grounded question-answering with citation tracking. Built as a deliberate engineering exercise with attention to ingestion resumability, retrieval performance, and answer trustworthiness.

The codebase follows a clean architecture / DDD-style layering:

- **Domain** — framework-agnostic entities and value objects
- **Application** — orchestration services using domain + infrastructure
- **Infrastructure** — concrete implementations (DB, HTTP clients, parsing, chunking)


## Tech Stack

| Concern | Choice | Why |
|---|---|---|
| API server | FastAPI + Uvicorn | Async-first, native Pydantic integration, lightweight |
| LLM (answer generation) | Anthropic Claude | Strong grounded-reasoning behavior; isolated behind a service for swap-out |
| Embeddings | OpenAI `text-embedding-3-small` | Best-in-class quality/cost; mixed providers were a deliberate choice over single-vendor lock-in |
| Database + vector store | PostgreSQL + pgvector | Single store for relational + vector data — one backup story, one failure surface, joinable with structured metadata for pre-filtered retrieval. Revisit if recall at scale becomes a bottleneck |
| ORM/driver | SQLAlchemy 2.0 + asyncpg / psycopg async pool | Async end-to-end matches the FastAPI runtime |
| Migrations | Alembic | Versioned, reversible migrations from day one — no "how is the schema reproducible?" question for the life of the project |
| HTML parsing | BeautifulSoup4 + lxml (custom parser) | Initially used `edgartools` but encountered breaking API changes between versions and limited control over section boundaries; rebuilt as ~200 lines of focused parsing code with no external library risk |
| Tokenization | tiktoken (cl100k_base) | Matches OpenAI embedding model tokenization for accurate chunk-size budgeting |
| CLI | Typer | Pydantic-like typing for command arguments; same mental model as FastAPI |


## Design Notes

- **Resumable ingestion** via the `Filing` status state machine — reruns pick up at the last completed phase instead of redoing work.
- **Status alone is insufficient to verify data integrity.** A filing can complete the state machine (status = `EMBEDDED`) while producing anomalously little data — e.g., a parser silently failing to detect sections, yielding one chunk where ~100 are expected. The `corpus-status` CLI command cross-checks status against chunk counts and embedded-chunk counts per filing, surfacing silent failures that status alone would mask. This caught the McDonald's parser issue and an earlier ingestion-limit bug during dogfooding; running `corpus-status` is now the first step of every dogfooding session.
- **Denormalized chunk metadata** (`ticker`, `filed_date`, `filing_type`, `section_path` repeated on every chunk row) trades storage for retrieval latency: a query like *"risk-factor chunks from AAPL filed after 2023"* applies the three metadata filters as a cheap bitmap index scan first, then runs the HNSW vector scan over only the filtered subset rather than the entire corpus.
- **TOC deduplication** in the parser (`filing_parser._locate_item_headings`) is a load-bearing detail — without it, sections capture empty table-of-contents entries instead of actual content, since EDGAR filings repeat "Item N" headings in both the TOC and the body.
- **Citation format** (`[TICKER FORM YEAR §Item]`) is human-readable and unambiguous across multi-filing corpora. The current implementation extracts cited tags from the answer by substring match but does not yet verify that each cited chunk *literally contains* the claimed text or numbers. Dogfooding surfaced a real failure case where this matters (see Limitations); per-claim citation verification is Phase 2 work.
- **Aggregate lifecycle vs read-side queries** - Repositories handle aggregate lifecycle (insert, update, find). Cross-aggregate read queries live under infrastructure/queries/ to keep write-side and read-side concerns visibly separate.
- Read-side queries return frozen dataclasses rather than dicts — moves field-name errors from runtime puzzles to static-checker / immediate-crash failures.


## Limitations

- **Citation precision not verified.** The LLM produces citations alongside its answer, but the system does not check whether each cited chunk literally contains the claimed text or numbers. Dogfooding surfaced a case where a confident answer cited a specific dollar figure to a chunk that did not contain that figure — the numbers were present in the corpus but in different chunks not selected by top-k retrieval. For an investment research tool, this is a critical failure mode: a plausible-looking citation may be unverifiable. Phase 2 will add a citation verification module that parses (claim, citation) pairs and confirms the cited chunk contains the claim's key terms before returning the answer.

- **Parser assumes explicit "Item N." section headings.** Discovered during dogfooding that some large-cap filers (confirmed: McDonald's) instead use business-friendly headings ("Business Summary", "Management's View of the Business") and rely solely on the SEC table of contents and anchor links to cross-reference Items. The current parser silently produces near-empty section maps on these filings. Phase 2 will rebuild parsing around TOC-anchor following, which handles both conventions uniformly.

- **Embedding-only retrieval is literal at a semantic level.** Vector search surfaces chunks lexically near the query, but does not synthesize across passages that discuss the same business concept under different vocabulary. Example: a query about "supply chain risks" on a filer who discusses the topic as "vendor dependencies" or "pharmaceutical procurement" may underperform. Phase 2 will evaluate hybrid search (BM25 + vector) and query expansion against an evaluation harness to measure whether either materially improves recall.

- **Numeric retrieval is unreliable on table-heavy filings.** Vector embedding quality is weak for chunks dominated by financial tables (numbers with minimal surrounding prose), and queries about specific financial metrics may miss the chunks containing the numbers. Confirmed during dogfooding with UnitedHealth's Optum Rx revenue: the relevant chunks (containing the $49,775M figure) were not retrieved in top-8 for the natural-language query, while similar queries on Apple (where the services revenue chunk has more surrounding prose context) surfaced the right chunk reliably. Phase 2 will evaluate chunking strategy changes (richer context around tables), hybrid search (BM25 + vector), and structured XBRL fact extraction against an evaluation harness.

- **Citation reliability degrades when retrieval misses the target chunk.** The LLM does not yet verify that cited chunks literally contain the claimed facts; when retrieval fails to surface the right chunk for a numeric question, the LLM may still confidently produce a plausible-looking citation. Three out of four tested cases showed citations were accurate when retrieval succeeded, so the immediate Phase 2 priority is fixing retrieval; citation verification is a secondary safety net.

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
