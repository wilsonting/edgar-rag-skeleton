# trading-agents

Two systems in one repository, joined at a single seam.

**An EDGAR RAG pipeline** (`app/`, everything outside `app/agent/trading/`) downloads SEC
filings (10-K, 10-Q, 8-K, 20-F), parses them into sections, chunks them, embeds the chunks
into Postgres + pgvector, and answers questions about them with citations that are checked
back against the source text.

**A multi-agent trading pipeline** (`app/agent/trading/`) takes one ticker and one analysis
date and produces a decision memo: a bull case, a bear case, a risk ledger, a verdict
(buy / sell / hold / unresolved), and an explicit list of what the run could *not* see.

The seam is the Fundamentals Analyst. It is not a new agent — it is a thin wrapper
(`app/agent/trading/infrastructure/fundamentals_port.py`) around the EDGAR research agent
that already existed, calling the same path as `python -m app.agent.researcher TICKER`.

About 16,300 lines of Python under `app/`, 612 test functions under `tests/`. The largest
single file is `infrastructure/debate_port.py` — which says something true about the
project: most of the code is not "call the model", it is **checking what the model said**.

---

## How a trading run flows

```
                    START
                      │
        ┌─────────────▼─────────────┐
        │  ANALYSTS (run in order)  │
        │  fundamentals ─ EDGAR RAG │
        │  technical    ─ prices    │
        │  news → sentiment         │
        └─────────────┬─────────────┘
        ┌─────────────▼─────────────┐
        │  DEBATE  (cycle)          │   bull ⇄ bear, 3 rounds = 6 turns
        └─────────────┬─────────────┘
                 debate_close        ← records WHY it stopped
        ┌─────────────▼─────────────┐
        │  RISK PANEL (cycle)       │   neutral → aggressive → conservative,
        └─────────────┬─────────────┘   3 rounds = 9 turns
                  risk_close         ← records WHY it stopped
        ┌─────────────▼─────────────┐
        │  SYNTHESIZER              │   Research Manager + Risk Judge,
        └─────────────┬─────────────┘   run 3 times, majority vote
                     END

   every edge above also has an "abort" branch → graceful_abort (budget/deadline)
```

Orchestrated with **LangGraph**, checkpointed to Postgres after every super-step, so a
killed run resumes at the node it stopped on rather than restarting. Each agent area is
split `domain/` (pure Pydantic types and rules — no network, no LLM, no clock),
`application/` (nodes and routers), `infrastructure/` (ports that talk to an LLM, a vendor
API, or disk).

Three design rules recur throughout:

- **The model writes prose; Python owns the facts.** Identifiers, counters and labels are
  Python-assigned, never trusted off the wire. Every figure in a debate turn must appear
  verbatim in the evidence pack — containment, not tolerance.
- **Termination is guaranteed three independent ways**: a pure router cap, a derived
  global `recursion_limit`, and runtime asserts at node entry.
- **A memo that hit a round cap must not read like a memo that resolved.** Every stage
  records its own blind spots into the memo's `data_gaps`.

---

## Prerequisites

- **Python 3.13** — pinned in `.python-version`. Not "3.13 or newer": `tiktoken==0.8.0`
  has no wheel for 3.14, and without the pin `uv` picks the newest interpreter it can find
  and falls back to a source build that needs a Rust toolchain.
- **[uv](https://docs.astral.sh/uv/)** — the lockfile (`uv.lock`) is committed
- **Docker** (for Postgres 16 + pgvector)
- API keys: **Anthropic** and/or **DeepSeek** (LLM roles), **OpenAI** (embeddings),
  **Finnhub** (company news, free tier is sufficient)
- An **SEC EDGAR User-Agent** string — SEC requires a contact identity on every request

## Setup

```bash
# 1. Dependencies
uv sync

# 2. Configuration — before the schema step, which reads POSTGRES_DATABASE_URL from it
cp .env.example .env    # then fill in the keys

# 3. Database (Postgres 16 + pgvector, exposed on host port 6432)
docker compose up -d

# 4. Schema
uv run alembic upgrade head
```

`.env.example` lists every variable with a comment; `.env` itself is gitignored.

> **`.env` is required, and three of its variables are read at import time.**
> `LLM_CLAUDE_MODEL` (via `model_for`), `LOOP_MAX_TURNS` and `MEMO_DIR`. A missing one
> stops the process before argument parsing with `MissingSetting: <NAME> is not set…`.
>
> **Precedence:** a variable already set in the environment wins over `.env`, which only
> fills in what is unset. Entry points (the API server, both CLIs, the researcher script)
> load `.env` before anything else; see `app/config.py`. The `scripts/` battery tools
> still load it with `override=True`, so inside them `.env` wins.

## Configuration

| Variable | Purpose |
|---|---|
| `POSTGRES_DATABASE_URL` | Postgres connection string (docker-compose exposes `localhost:6432`, db `rag`, user `postgres`) |
| `TRADING_CHECKPOINT_DB_URI` | Connection string for the LangGraph checkpointer |
| `OPENAI_API_KEY` | Embeddings |
| `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` | LLM roles, per provider actually used |
| `FINNHUB_API_KEY` | Company news for the news/sentiment leg |
| `EDGAR_USER_AGENT`, `EDGAR_IDENTITY` | SEC contact identity — required by EDGAR |
| `EDGAR_CACHE_DIR` | Where downloaded filing HTML is cached |
| `EMBEDDING_MODEL`, `OPENAI_MODEL` | Embedding model (`text-embedding-3-small`, 1536 dims) |
| `LOOP_MAX_TURNS` | **Required.** Turn cap for the EDGAR research agent loop |
| `MEMO_DIR` | **Required.** Vault directory for saved artifacts, resolved relative to `$HOME` |

### LLM roles

Provider is inferred from the model id's prefix — `claude-*` → Anthropic, `deepseek*` →
DeepSeek, `gpt-*`/`o1`/`o3`/`o4` → OpenAI — so one run can mix providers per role without a
global switch. Ten roles each read their own variable, all falling back to
`LLM_CLAUDE_MODEL`, which is the only required one:

| Role | Variable |
|---|---|
| research agent loop, and the fundamentals node that runs it | `LLM_CLAUDE_MODEL` |
| answer generation behind `POST /ask` | `LLM_ANSWER_MODEL` |
| query decomposition | `LLM_DECOMPOSER_MODEL` |
| financial-metric extraction | `LLM_EXTRACTION_MODEL` |
| news digest | `TRADING_NEWS_DIGEST_MODEL` |
| technical-indicator interpretation | `TRADING_TECHNICAL_MODEL` |
| bull/bear debate turns | `TRADING_DEBATE_MODEL` |
| risk panel personas | `TRADING_RISK_MODEL` |
| research manager synthesis | `TRADING_RESEARCH_MANAGER_MODEL` |
| risk judge, final verdict | `TRADING_RISK_JUDGE_MODEL` |

Print what a run will actually use — role, model, provider, and whether it is priced:

```bash
uv run python -m app.infrastructure.llm.models
```

Optional overrides: `LLM_PROVIDER` + `LLM_BASE_URL` + `LLM_API_KEY` bypass prefix inference
wholesale (for a gateway whose model ids carry no useful prefix); `LLM_PRICING_OVERRIDES`
(JSON) merges over the built-in token pricing table, so a repriced model is a config change
rather than a code change.

---

## Running it

### 1. Ingest filings into the corpus

```bash
uv run python -m app.cli ingest AVGO --type 10-K --limit 4 --since 2023
```

Runs the full pipeline for one ticker: discover → download → parse → chunk → embed. The
`Filing` status state machine makes this resumable — a rerun picks up at the last completed
phase.

Then verify what actually landed. Status alone is not integrity: a filing can reach
`EMBEDDED` while a parser silently produced one chunk where ~100 were expected.

```bash
uv run python -m app.cli corpus-status --ticker AVGO --verbose
```

This cross-check is the first step of every session, not an occasional audit.

### 2. Start the API server

```bash
uv run uvicorn app.main:app --reload
```

The agent tool layer calls this over HTTP at a hardcoded `http://localhost:8000`
(`app/agent/tools.py`), so **the server must be running on port 8000 before any agent or
trading run** — the fundamentals leg goes through it.

| Endpoint | Purpose |
|---|---|
| `POST /ask` | Grounded Q&A — decompose, hybrid-retrieve, answer with citations, verify |
| `POST /extract` | Extract structured financial metrics for one ticker/period |
| `POST /ingest` | Run the ingestion pipeline for one ticker |
| `POST /latest-filings` | Compare EDGAR's recent filings against what is in the corpus |
| `GET /corpus-status` | Same cross-check as the CLI command |
| `POST /news-assess` | Assess a headline against the watchlist thesis and the corpus |
| `POST /trading/analyze` | Run the trading graph for one ticker over HTTP |

### 3. Run the EDGAR research agent

```bash
uv run python -m app.agent.researcher AVGO                    # full research checklist
uv run python -m app.agent.researcher AVGO --news "AVGO announces 10B buyback"
```

Tool traces go to stderr, the memo to stdout — `2>/dev/null` yields a clean memo.

**News mode reads `watchlist.yaml`** from the repo root — per-ticker `thesis`,
`key_metrics` and `risks_watching` that the headline is assessed against, currently six
tickers (AVGO, ACN, NFLX, FIG, ASML, MSFT). A ticker that is not listed falls back to
generic framing, and so does *every* ticker if the file cannot be found: `_load_watchlist`
returns an empty list and warns on stderr rather than failing. The path is relative to the
working directory, so run these commands from the repo root — from anywhere else the
assessment still produces a confident-looking verdict, with none of the thesis context it
claims to be checking against.

### 4. Run the trading pipeline

```bash
uv run python -m app.agent.trading.interface.cli AVGO --as-of 2026-08-29 --max-usd 0.75
```

| Flag | Meaning |
|---|---|
| `--as-of DATE` | Analysis date. All news and price data are bounded at or before it. Defaults to today — the one `date.today()` in the trading CLI, at the argument parser. |
| `--only ANALYST` | Run only this analyst (`fundamentals`, `technical`, `news`); repeat to select several. The synthesizer still runs and records the others as data gaps. |
| `--max-usd FLOAT` | Hard per-run cost cap. Default `0.75`. |
| `--wall-clock-timeout-s FLOAT` | Hard per-run deadline in seconds. Default `1800`. |
| `--thread-id ID` | Checkpoint thread. Defaults to `trading-<TICKER>`; a `--only` subset gets its own suffix so a narrower graph never resumes a full run's checkpoint. |

**Resumes inherit the checkpoint's budget and deadline, never the flags on the current
command line.** The deadline is an absolute instant fixed when the run first started, not a
fresh window per attempt. The CLI refuses a resume whose inherited deadline has already
passed, and says so — because the run-level guards can only fire between nodes, so a doomed
resume would otherwise pay for the whole fundamentals stage and then abort with no memo.
To change a budget, start a fresh thread.

## What a run produces

Markdown artifacts in a dated vault folder under `$HOME/$MEMO_DIR` — fundamentals memo,
technical report, sentiment report, debate transcript, risk transcript, decision memo — a
JSON memo on stdout, and one `run_summary` line appended to `docs/cost-log.jsonl`.

Two fields in the memo are worth reading correctly:

- **`evidence_quality`** is what the run had to work with, *not* how likely the verdict is
  right. It was renamed from `confidence` for exactly that reason: measured across
  identical re-runs it moved 0.04, which is more than inverting a company's entire
  fundamentals picture moved it. Treat it as an input-quality measure.
- **`verdict`** may be `unresolved`. The Research Manager + Risk Judge pair runs three times
  over independently sampled risk panels and the verdict is the majority; no majority
  reports `unresolved` rather than picking one. No single model call can declare
  non-resolution — the tool schema offers only buy/sell/hold.

## Cost and safety rails

Budget and deadline are set once at the CLI boundary and never mutated. Every router is
wrapped with a guard, so a breach routes to `graceful_abort` from anywhere in the graph.
Measured cost is **$0.25/run on deepseek-v4-flash, $0.69–$1.04/run on Haiku 4.5**, over
roughly 560–920 seconds — wall clock binds before cost.

Numeric reliability rests on a verification chain built over six iterations, each closing a
channel and relocating the failure to a narrower one:

```
answer_question   grounding prompt: cite everything, arithmetic on in-context figures only
/ask              citation_verifier: literals + quotes vs retrieved chunks
ask_edgar         appends an in-band WARNING for unverified figures
calculate         schema-required inputs[] → reject unit multipliers →
                  reject undeclared literals → reject inputs absent from provenance
agent loop        memo_verifier re-runs the citation verifier over the final memo
```

The load-bearing lesson: **prose rules were followed procedurally and circumvented four
consecutive times**, while the two fixes that stuck were a schema change and a mechanical
corpus check — both enforced outside the model's discretion. A calculator launders invented
inputs into tool-authorized outputs, and declaring a source is not provenance; the check
that held was that the value must have literally appeared in a tool output during this run.

## Tests

```bash
uv run pytest -q -rs
```

612 test functions (702 cases after parametrization), 464 of them under
`tests/agent/trading/`. The whole suite runs in **under 15 seconds**, reaches no provider
and spends nothing — `tests/conftest.py` injects a placeholder credential for every
non-Anthropic provider, since the client layer validates keys at construction rather than
at first call.

Four environment variables are enough to run it: `LLM_CLAUDE_MODEL`, `LOOP_MAX_TURNS`,
`MEMO_DIR`, and a Postgres URL. Without a database the suite still passes, skipping five
checkpoint tests — but those five are the ones guarding a *delayed* failure, where an
unregistered domain type serializes fine in-process and breaks only when another process
reads the checkpoint back. Skipped, they read as passing, so **CI runs them against a real
pgvector service** (`.github/workflows/tests.yml`).

The pure-function design is what makes this affordable: routers, guards and domain rules
are exhaustively testable in milliseconds at zero API cost.

## Repository layout

```
app/
  main.py                    FastAPI server
  cli.py                     ingest / corpus-status / eval / extract-metrics
  domain/ application/ infrastructure/     EDGAR RAG pipeline
  agent/
    researcher.py prompts.py tools.py      EDGAR research agent (raw tool-use, no framework)
    trading/
      domain/                pure types and rules
      application/           nodes, routers, guards
      infrastructure/        LLM ports, graph, checkpointer, cost log
  infrastructure/llm/        provider routing, model table, pricing
.github/workflows/tests.yml   CI: pytest against a real pgvector service
.env.example                 every variable, commented
eval/                        retrieval evaluation harness
migrations/                  Alembic
scripts/                     validation batteries and probes
tests/
```

## Known limitations

Named, not hidden. The significant ones:

- **A historical run bounds retrieval, not the model's priors.** `as_of_date` now reaches
  every leg: filing retrieval is capped at the analysis date (`filed_before` on every
  `ask_edgar`, `check_latest_filings` and `extract_metrics` call), and the memo says so.
  What no bound can reach is what the model already knows about how the period turned out.
- **The budget is checked on edges, not inside nodes.** The synthesizer is one node making
  at least 24 model calls, so the documented "overshoot by at most one call" bound is wrong
  for the most expensive node in the pipeline.
- **The fundamentals cache keys on ticker alone** and is written on every real run but read
  only under `MOCK_FUNDAMENTALS=1` — one environment variable away from pairing an August
  memo with a March analysis date.
- **Parser assumes explicit "Item N." headings.** Filers using business-friendly headings
  (confirmed: McDonald's) produce near-empty section maps.
- **Incorporation-by-reference filers fail.** IBM's 10-K points Items 7/7A/8 at a separate
  exhibit; the pipeline does not follow exhibit references.
- **Prolific filers are truncated.** `EdgarClient.list_filings()` does not follow SEC
  submissions pagination, so JPM is limited to its most recent 10-K.
- **Parser fixes do not retroactively repair ingested filings.** Recovery requires deleting
  `sections` *and* `chunks` and resetting status to `discovered`; `corpus-status` checks
  chunk counts, not section completeness, so this staleness is invisible.
- **The verifier answers "does this literal exist in the source", not "is this claim true".**
  Fabricated causation has no literal to check, and a real figure attributed to the wrong
  fiscal year passes.
- **A missing `watchlist.yaml` degrades silently.** News assessment falls back to generic
  framing with only an stderr warning, so the failure looks like a completed run. The path
  is resolved against the working directory, not the module.

**Operating envelope for readers of the output:** retrieved and cited figures are reliable;
computed figures are reliable when produced by `calculate`; period labels and causal
explanations warrant a spot-check.

## Further reading

`docs/` holds the engineering journals — `architecture.md`, `tutorial.md` (a full code
walkthrough), `trading-agent-known-gaps.md` (32 dated sections), and the validation
batteries under `docs/validation/`. Two things under `docs/` are gitignored and stay local:
`cost-log.jsonl`, which every run appends to, and the raw `*.stdout`/`*.stderr` run output
under `docs/validation/`. If
you are about to change something, read the dated section covering it first: most surprises
in this codebase have already been surprising once and were written down.
