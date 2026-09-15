# Code review: trading-agents

**Reviewed:** `main` at `617387f` (after #100, the memo verdict-consistency flag), 2026-09-12.
**Scope:** `app/` (~11.5k lines), `tests/` (~14.8k lines), `eval/`, `scripts/`, `migrations/`, CI, packaging, docs.
**Method:** read the HTTP API, the CLI, the research agent and its tool layer, the trading graph, nodes, routers and guards, the fundamentals/technical/news/debate/risk/synthesis ports, the budget and cost path, retrieval and the SQL repositories, ingestion, EDGAR, parsing, chunking, the verifiers, provider routing and config. Ran the full suite, an AST unused-import pass, and an env-var cross-check of `.env.example` against what the code actually reads. Where a claim needed proof I reproduced it (marked **[verified]**); otherwise **[from code]**.

**Suite at review time:** `881 passed, 1 skipped in 3.49s` (the skip is `test_manifest_merge.py`, which needs a `docs/validation/` file absent from this checkout).

---

## Summary

The core is in good shape. Since the previous review (deleted from the tree at `4e966a7`; recoverable with `git show 0e8e80d:docs/code_review.md`) **all three High and all ten Medium findings shipped** across PRs #82–#95: the API and CLI now share one run lifecycle, the budget guard reaches inside the fundamentals loop and the synthesizer, BM25 actually returns hits, per-run state moved to ContextVars, number matching is consolidated, ingestion retries FAILED filings idempotently, and config loads `.env` exactly once per entry point. That pass left its **Low section entirely unaddressed**, and this review confirms every item in it is still open.

What this review adds is mostly at the edges the last one flagged and in the places the last one did not reach:

- **A shipped CLI command is broken** — `extract-metrics` raises `AttributeError` on its first filing.
- **The fundamentals leg still reads the wall clock**, so a historical `--as-of` run has one unbounded analyst. The README names this; nothing in the code does.
- **The eval harness measures a retrieval path production does not use**, so the numbers that justified the BM25 work do not describe `/ask`.
- **The pre-refactor PDF pipeline is still in `app/`**, along with a Python 2.7 `.pyc` committed to git.
- **Three ports carry a byte-identical copy of the same six helpers.**

Nothing here is a security exposure: CORS is scoped, the spending endpoints take an optional shared secret, tickers are validated at every entry point including the vault writer, and Postgres is bound to loopback. Those were the previous review's #4 and they held.

### Priority list

| # | Severity | Finding | Where |
|---|---|---|---|
| 1 | High | `extract-metrics` CLI crashes — wrong `FinancialMetrics` type reaches the repository | `cli.py:416-424`, `metrics_repo.py` |
| 2 | High | `as_of_date` never reaches the fundamentals leg — lookahead in every historical run | `nodes.py:103`, `fundamentals_port.py:101` |
| 3 | Medium | Eval never exercises `retrieve_full`, the path `/ask` uses | `eval/runner.py:136-151` |
| 4 | Medium | The forced-memo turn is billed at full price — no tools, no cache breakpoint | `researcher.py:683-690` |
| 5 | Medium | Sub-query results are merged by max RRF, discarding cross-query agreement | `retrieval_service.py:279-285` |
| 6 | Medium | Dead pre-refactor PDF pipeline still in `app/`; Python 2.7 `.pyc` tracked in git | `app/ingest.py`, `app/retrieve.py`, `app/db_postgres.py`, `app/db.py`, `app/chunk.py`, `app/__init__.pyc` |
| 7 | Medium | Seven direct `os.environ[...]` / `os.getenv` reads bypass `require_env` | `main.py`, `cli.py`, `db.py`, `checkpointer.py` |
| 8 | Medium | The cost log has two hardcoded relative paths and no rotation | `researcher.py:390`, `cost_log.py:29` |
| 9 | Medium | `_accumulate` + five more helpers are copied across three ports | debate/risk/synthesis ports |
| 10 | Medium | `use_hybrid` is threaded through four call sites and read by nothing | `retrieval_service.py:41` |
| 11 | Medium | Price-fetch failures are silent; `content[0]` is unguarded; `researcher --test` crashes | price port, `llm.py`, `researcher.py:748` |
| 12 | Low | Unused imports, unused models, `USE_STUBS` scaffolding, stale docstrings | many |
| 13 | Low | Duplicate/diverged docs, stale README limitation, undeclared dependencies | root vs `docs/` |
| 14 | Low | No linter, no type checker, no coverage of any CLI command in CI | `.github/workflows/tests.yml` |

---

## High

### 1. `extract-metrics` crashes on its first filing [verified]

`app/cli.py:416-424` runs extraction and hands the result straight to the repository:

```python
metrics = await extractor.extract(chunks, ticker, f.fiscal_period, f.filing_type, f.filed_date)
await metrics_repo.upsert(metrics)
```

`MetricsExtractor.extract` returns the **Pydantic** `FinancialMetrics` from `app/application/extraction_service.py` — revenue, margins, confidence, reasoning. `MetricsRepository.upsert` expects the **dataclass** `FinancialMetrics` defined in `app/infrastructure/repositories/metrics_repo.py:11`, and its first statement is `metrics.ticker.upper()`.

```
CONFIRMED AttributeError: 'FinancialMetrics' object has no attribute 'ticker'
has source_citations? False
```

`ticker`, `fiscal_period`, `filing_type`, `filed_date` and `source_citations` are all absent. `POST /extract` gets this right (`main.py:354-370` builds a `MetricsRow` explicitly); the CLI path never did.

Two things make it invisible. `metrics_repo.py:5` imports the extraction-service `FinancialMetrics` and then shadows it with its own dataclass on line 11 — the shadowed import is almost certainly where the confusion started. And no test touches any `app.cli` command: `grep -rn "app.cli" tests/` returns nothing.

`_run_extract_metrics` also never calls `init_pool()` (every other CLI coroutine does); `get_connection` self-heals, so the pool is opened lazily and never closed, but the asymmetry is worth removing.

**Fix:** have the two identically-named types stop colliding — rename the repository row to `FinancialMetricsRow`, delete the shadowed import — and build it in one shared helper both `main.extract` and `cli._run_extract_metrics` call. Add a test that runs the command against stubbed repositories.

### 2. `as_of_date` never reaches the fundamentals leg [from code]

Every other data source is bounded at the analysis date and asserts it. `price_data_port._bound_to_as_of` drops post-`as_of` bars inside both vendor helpers; `news_node` and `synthesizer_node` refuse to run without `as_of_date` rather than defaulting to today; `TradingState.as_of_date`'s own comment says a node calling `date.today()` internally "makes probe-date runs impossible to verify".

The fundamentals leg does exactly that:

```python
# nodes.py:103 — as_of_date is in state, and is not passed
report = await get_fundamentals_report(state["ticker"], run_id=..., budget=..., prior_events=...)

# fundamentals_port.py:101
today = date.today()
task = f"Today's date is {today.isoformat()}. Run the full research checklist for {ticker}."
```

`get_fundamentals_report` has no `as_of` parameter at all. `today` also becomes `FundamentalsReport.generated_at`. So a run invoked with `--as-of 2026-03-01` produces a memo whose news and prices stop at March and whose fundamentals research reads whatever is in the corpus and on EDGAR today — and the memo does not say so. That is lookahead bias in the one analyst whose output carries the most weight.

Two related hazards in the same file:

- `_cache_path(ticker)` keys the fundamentals cache on **ticker alone**. It is written on every real run (`fundamentals_port.py:153-154`) and read only under `MOCK_FUNDAMENTALS=1`. One environment variable pairs a months-old memo with any analysis date.
- The cache directory is `app/agent/trading/.fundamentals_cache` — inside the source tree.

**Fix:** thread `as_of` from `TradingState` into `get_fundamentals_report`, put it in the task prompt, and set `generated_at` from it. Key the cache on `(ticker, as_of)`, and either move it under `MEMO_DIR` or gate the *write* on `MOCK_FUNDAMENTALS` too. Until the agent's retrieval can be bounded by filing date, have the port add an explicit data gap saying the fundamentals leg is not point-in-time.

---

## Medium

### 3. The eval harness measures a path production does not use [from code]

`eval/runner.py:136-151` picks exactly one of three modes:

| flag | method called | decomposition | hybrid |
|---|---|---|---|
| `use_hybrid` | `retrieve_hybrid` | no | yes |
| `use_decomposition` | `retrieve_with_decomposition` | yes | no (vector only) |
| neither | `retrieve_by_embedding` | no | no |

`POST /ask` calls **`retrieve_full`** — decomposition *and* hybrid — and `retrieve_full` is called by nothing in `eval/`. `retrieve_with_decomposition` is called by nothing *but* `eval/runner.py`. So the harness has a method kept alive only for measurement, and the method that serves every agent question is never measured.

This matters because the BM25 fix (#84) was justified by a measurement, and the measured configuration is not the shipped one: fusion behaves differently when its inputs are sub-queries.

The gold sets compound it — they are keyed by serial chunk ids, which change on every re-ingest, `eval/test_set.yaml` is gitignored, and eval is not in CI.

**Fix:** add a `full` mode that calls `retrieve_full`, make it the default, and delete `retrieve_with_decomposition` once nothing needs it. Key gold sets by `(accession, section_path, content hash)`.

### 4. The forced-memo turn cannot read the prompt cache [from code]

The loop is careful about caching: `_roll_cache_breakpoint` moves a single breakpoint to the last block each turn, and the system block carries its own `cache_control`, so each turn re-reads tools + system + history at ~0.1×.

The final call does neither (`researcher.py:683-690`):

```python
response = await client.messages.create(
    model=AGENT_MODEL,
    max_tokens=AGENT_MAX_TOKENS,
    system=system_prompt,      # plain string — no cache_control breakpoint
    messages=messages,         # no tools=TOOLS
)
```

The cached prefix is tools → system → messages, so dropping `tools` invalidates it from the first byte, and with no breakpoint anywhere there is nothing to read from cache regardless. This is the turn that sends the *entire* accumulated conversation, and `tools.py:27-45` records that Phase 9 measured 2 of 3 fundamentals runs ending on exactly this path.

The `max_tokens` continuation call inside the loop (`researcher.py:604-616`) does keep `cache_control`, and it also omits `tools` — same invalidation, smaller blast radius.

**Fix:** pass `tools=TOOLS` and the structured `system=[{... cache_control ...}]` block on both calls, and call `_roll_cache_breakpoint(messages)` before each. Verify against the `cache_read_ratio` already recorded in each `run_summary` line.

### 5. Sub-query results merge by max, not by agreement [from code]

`retrieve_full` fuses within a sub-query with RRF, then merges across sub-queries with a max (`retrieval_service.py:279-285`, and identically at `:112-116`):

```python
if existing is None or chunk.similarity > existing.similarity:
    all_chunks[chunk.chunk.id] = chunk
```

RRF scores are rank-derived and comparable *within* a ranking, so `max` amounts to "best rank this chunk reached in any sub-query". A chunk that every sub-query ranked 3rd scores the same as one that a single sub-query ranked 3rd and the others missed entirely — the multi-query agreement signal that is the whole reason to decompose is discarded. Summing the per-sub-query RRF contributions is the standard treatment and is a two-line change.

`main.gather_extraction_chunks` (`main.py:373-382`) has the same shape over `METRIC_QUERIES`, and additionally runs its four queries **sequentially**, as does `RetrievalService.retrieve_for_extraction:233-236` — both missed by the #93 parallelisation pass.

### 6. The pre-refactor PDF pipeline is still shipped, and a Python 2 `.pyc` is in git [verified]

`app/ingest.py`, `app/retrieve.py`, `app/db_postgres.py`, `app/db.py` and `app/chunk.py` are the original tutorial pipeline. Nothing at runtime imports them — `tests/test_config.py:25` already lists two of them as `_DEAD_MODULES` so the "only `app/config.py` loads dotenv" guard can pass, which is the tell: **the dead modules are the only reason that exemption exists**. Both still call `load_dotenv(override=True)` at import, the exact pattern `app/config.py` was written to eliminate. `app/ingest.py` writes to a `chunks(source, chunk_index, …)` schema that no longer exists.

`app/__init__.pyc` is tracked (`git ls-files | grep pyc`), 102 bytes, magic `03f3 0d0a` — CPython **2.7** bytecode. `.gitignore` has `__pycache__/` but no `*.pyc`.

Deleting the five modules also removes the last reader of `EMBEDDING_MODEL` outside `EmbeddingService`, and lets `_DEAD_MODULES` come out of `test_config.py`.

### 7. Seven env reads bypass `require_env` [verified]

`app/config.py` exists to turn a missing setting into a message naming `.env.example`. Seven reads still raise a bare `KeyError` or produce a confusing downstream failure:

| file:line | variable | failure |
|---|---|---|
| `main.py:410`, `main.py:440` | `EDGAR_USER_AGENT` | `KeyError` → HTTP 500 |
| `cli.py:169`, `cli.py:361` | `EDGAR_USER_AGENT` | `KeyError` traceback |
| `repositories/db.py:13` | `POSTGRES_DATABASE_URL` | `KeyError` |
| `db_postgres.py:8` | `POSTGRES_DATABASE_URL` | `KeyError` at import (dead module — see #6) |
| `checkpointer.py:22` | `TRADING_CHECKPOINT_DB_URI` | `os.getenv` → `None` → `AsyncConnectionPool(conninfo=None)` |

`AGENT_TOOL_CONCURRENCY` is read by the code and documented nowhere — it is the only variable missing from `.env.example` (the env cross-check found no others; the "documented but unread" names are all resolved dynamically through `models.ROLES`).

### 8. The cost log: two hardcoded paths, no rotation [from code]

`Path("docs/cost-log.jsonl")` is written literally in two places — `researcher.log_cost:390` and `cost_log._COST_LOG_PATH:29` — with no shared constant and no env override. Both are relative to the process working directory, so a run started anywhere but the repo root silently creates a new `docs/` there and writes a log nothing reconciles against.

The file is append-only and unbounded: 3,177 lines / 1.0 MB today. `_disk_logged_events` reads and JSON-parses the whole file once per run summary, filtering by `run_id`.

**Fix:** one `COST_LOG_PATH` constant resolved from the repo root (or `COST_LOG_PATH` env), imported by both. Rotate by month (`cost-log-2026-09.jsonl`) and have the reconciliation read only the current file plus the previous one.

### 9. Six helpers copied verbatim across three ports [verified]

`_accumulate` is byte-identical in `debate_port.py:1219`, `risk_port.py:418` and `synthesis_port.py:691`. Alongside it each port carries its own `_tool_block`, `_extract`, `_CORRECTION`, `_retry_messages` and `_assert_within_budget`, plus `_maybe_crash` with its own `_CRASH_AT`/`_CRASH_WHEN` pair. `news_digest_port` has a fourth `_assert_within_budget`.

The three ports are 1,581 + 766 + 1,002 lines. A shared "forced tool call with one schema retry, cost accounting, a crash hook and a spend ceiling" helper would remove a few hundred lines, and — more valuable — would give the four budget ceilings one implementation to fix rather than four to keep in step.

### 10. `use_hybrid` is a knob connected to nothing [verified]

`RetrievalService.__init__` stores `self.use_hybrid` (`:41`) and no method ever reads it. Callers choose hybrid by *calling* `retrieve_hybrid`/`retrieve_full` directly. The parameter is nonetheless passed from `main.py:262`, `main.py:335`, `cli.py:402`, `eval/runner.py:130` and one test — and in `eval/runner.py` the same flag *does* steer behaviour via an `if`, so the two meanings sit one frame apart.

This is the precise trap `models.py`'s docstring describes ("two variables that no code read at all — so changing them looked like it worked and did nothing"), recurring as a constructor argument.

### 11. Three small failures that read as success [from code]

- **Silent price fetches.** `_try_yfinance:124` and `_try_finnhub:156` catch bare `Exception` and `return None`, with no logging. A rate limit, an auth failure and "this ticker has no data" are indistinguishable — and the fallback chain means a broken primary vendor looks like a normal secondary hit.
- **Unguarded `content[0]`.** `llm.py:69` and `query_decomposer.py:160` read `resp.content[0].text` with no length check. The OpenAI-compat adapter returns no content blocks when the provider sends no text (output cut at the token limit is the common case), which surfaces as `IndexError` → HTTP 500 from `/ask`.
- **`researcher --test` crashes.** `main()` sets `mode` on the `--news` and ticker branches only; the `--test` branch leaves it unbound, and line 748 reads `if mode != "test"`. `python -m app.agent.researcher --test` raises `UnboundLocalError` after the run completes. The docstring advertises the flag at the top of the file.

---

## Low

**Dead code and leftovers** (all verified by an AST pass; `from __future__ import annotations` false positives excluded)

- `app/agent/trading/interface/cli.py:20` imports `RECURSION_LIMIT`, `_describe_stale_budget` and `_humanize` from `runner` and uses none of them — leftovers from the #82 extraction.
- `runner._describe_stale_budget` takes `wall_clock_timeout_s` and never uses it.
- `main.py:164` `FinancialMetricsResponse` — referenced nowhere.
- `main.py:36` and `main.py:43` both import `format_citation_tag`.
- `metrics_repo.py:6` `from sqlalchemy.dialects.postgresql import insert` — unused, and the only SQLAlchemy import in the repository layer.
- `metrics_repo.py:5` imports `FinancialMetrics` and line 11 shadows it (see #1).
- `edgar/client.py:6` `from sqlite3 import connect`; `edgar/ticker_resolver.py:3` `date`; `app/llm.py:5` `Chunk, Chunks`; `app/domain/chunk.py:2` `Field`; `section_repo.py:1` `Json`; `app/cli.py:8` `asdict`; `debate_port.py:47` `create_with_temperature_fallback`; `risk_port.py:38` `RISK_MAX_TURNS`; `eval/runner.py:9,14` `MetricsExtractor`, `init_pool`, `close_pool`.
- `tools.py:487` `USE_STUBS = False`, `_stub()`, and the "STEP 2 / Un-stub by setting…" comments describe a wiring step finished long ago.
- `MetricsRepository(session_factory=None)` — the parameter is stored and never read, and every call site passes `None`.
- `Filing.transition_to` / `fail` are now called (ingestion retry, #89), so that item is closed.
- A stale git worktree sits at `.claude/worktrees/inspiring-kilby-ec9f13` (detached at `c35b498`, last touched 2026-08-29) with a full duplicate of `app/` and `tests/`. It distorts any repo-wide grep.

**Small correctness notes**

- `calculate` returns a bare number. `_scale_by_value` normalises declared inputs to ones, so a difference of two thousands-denominated figures comes back 1000× the source table's scale with nothing saying so. `_scale_by_value` also keys by float value, so two inputs sharing a value but not a unit silently collide.
- `chunk_repo.search_by_embedding`'s docstring still claims filters "execute as a Bitmap Index Scan BEFORE the HNSW similarity scan". pgvector does not combine indexes that way; with a selective filter it either sorts the filtered rows exactly or filters after the approximate scan, and the second can return fewer than `k` rows as the index grows. Consider `hnsw.iterative_scan` (pgvector ≥ 0.8) or exact search when a ticker filter is present.
- `edgar/client.py:94,97` calls `asyncio.get_event_loop()` inside a coroutine; use `get_running_loop()`.
- `list_filings` reads only `filings.recent` and ignores `filings.files`, so prolific filers are silently truncated (the README names this; the code does not).
- `memo_verifier._appended` emits `["", "---"] + verdict_lines` while the report path emits `["", "---", ""] + verdict_lines` — one extra blank line, two spellings of the same separator.
- `_sample_additional_risk_panel` runs a full 9-turn panel with no budget check between turns; the run-level check happens only *before* each sample. The port's own `NodeBudgetExceeded` is the only thing bounding a runaway inside one sample.
- `tools.py:24` `API_BASE = "http://localhost:8000"` is a module constant with no env override, and every tool call constructs a fresh `httpx.AsyncClient` (`tools.py:601`) — no connection reuse, which #93 fixed everywhere else.

**Dependencies**

- `pyyaml` is imported by `researcher.py`, `eval/runner.py` and `eval/extract_runner.py`; `pandas` by `technical_indicators.py` and `price_data_port.py`. Neither is declared in `pyproject.toml` — both arrive transitively (`uvicorn[standard]`, `yfinance`). A transitive bump can remove either.

**Documentation**

- `architecture.md` exists at the repo root (644 lines) and in `docs/` (711 lines), and the two have **diverged**. `trading-agent-known-gaps.md` and `watchlist.yaml` are byte-identical duplicates in both places — and `researcher.WATCHLIST_PATH` reads the **root** copy, so `docs/watchlist.yaml` is a decoy that can drift without anything noticing.
- The README's "Known limitations" still says *"The budget is checked on edges, not inside nodes… the documented 'overshoot by at most one call' bound is wrong for the most expensive node"*. PR #83 fixed that; the entry is now false.
- README test counts are stale: "612 test functions (702 cases after parametrization)" vs. 881 passing today.
- `app/llm.py`'s `SYSTEM_PROMPT`, behind every `/ask`, contains "the enough information", "fall back on generate knowledge", "No premable", and an unclosed quote at `- Be concise, No premable. no "Based on the provided context,`.
- `docs/tutorial.md` is mode `600` where every other tracked file is `644`.

**CI and tooling**

- The workflow runs `pytest` only. No linter, no formatter, no type checker — every unused import in #12 would have been caught for free by `ruff check`.
- No CLI command has a test (`app/cli.py` and `app/agent/trading/interface/cli.py` between them are 709 lines). #1 is the direct consequence.
- `tests/agent/trading/test_manifest_merge.py:82` skips on a `docs/validation/` file that is gitignored, so it skips in CI too and reads as passing.

---

## Suggested order of work

1. **Fix what is broken (#1, #11).** `extract-metrics`, `researcher --test`, the two `content[0]` reads. Small, self-contained, each with a test.
2. **Close the lookahead hole (#2).** This is the one finding that changes what the product's output *means*.
3. **Make the measurements describe production (#3, #4, #5).** Eval on `retrieve_full` first, because #4 and #5 both want before/after numbers.
4. **Delete what is dead (#6, #10, #12).** Largest reduction in reading surface per unit of risk, and it unblocks the `_DEAD_MODULES` exemption in `test_config.py`.
5. **Consolidate (#7, #8, #9).** `require_env` everywhere, one cost-log path, one structured-call helper.
6. **Add the tooling that would have found half of this (#14).**

---

## Fix checklist

`[x]` = done and tested, `[~]` = in progress, `[ ]` = not started. Each High item ships as its own PR.

**Status, 2026-09-12:** every High and Medium item is addressed across PRs #102–#107, which stack in that order (#102 → #103 → #104 → #105 → #106 → #107). Suite at the tip: 951 passed, 1 skipped. Six sub-items are left open and each says below why — all six need either the gitignored eval corpus or a live paid run.

### High

**#1 `extract-metrics` crashes**: PR #102
- [x] Rename the repository dataclass to `FinancialMetricsRow` and delete the shadowed `from app.application.extraction_service import FinancialMetrics` in `metrics_repo.py`
- [x] One shared `build_metrics_row(extracted, ticker, fiscal_period, filing_type, filed_date, chunks)` used by both `main.extract` and `cli._run_extract_metrics`
- [x] `_run_extract_metrics` opens and closes the pool like every other CLI coroutine
- [x] Test: `extract-metrics` end to end against stubbed extractor and repository, asserting the row's `ticker`/`fiscal_period`/`source_citations`
- [x] Test: at least one smoke test per `app.cli` command, so no shipped command is untested again

**#2 `as_of_date` does not reach the fundamentals leg**: PR #103
- [x] `get_fundamentals_report(ticker, as_of, ...)` — required, not defaulted
- [x] `fundamentals_node` passes `state["as_of_date"]`; the port raises if it is missing, as `news_node`/`synthesizer_node` do
- [x] The task prompt states the analysis date; `FundamentalsReport.generated_at` is set from it, not `date.today()`
- [x] Fundamentals cache keyed on `(ticker, as_of)`; writes gated on `MOCK_FUNDAMENTALS` or moved under `MEMO_DIR`
- [x] Until retrieval can be bounded by filing date, the port emits an explicit data gap naming the fundamentals leg as not point-in-time
- [x] Test: a run with `as_of` in the past reaches the port with that date and never calls `date.today()`
- [x] Update the README's "Known limitations" entry to match what ships

### Medium

**#3 Eval measures the production path**: PR #104
- [x] `eval/runner.py` gains a `full` mode calling `retrieve_full`; make it the default
- [ ] Re-run the 51-query BM25 comparison under `full` and record the numbers beside the #84 ones — **not done:** needs the gitignored `eval/test_set.yaml` and spends decomposer calls per question
- [x] Delete `retrieve_with_decomposition` once nothing calls it
- [ ] Key gold sets by `(accession, section_path, content hash)` instead of serial chunk id — **not done:** the test set is gitignored and absent from this checkout, so the migration cannot be written against real data
- [ ] Commit a small `eval/test_set.yaml` (or a fixture subset) so eval can run in CI — **not done:** same reason. The harness itself is now unit-tested (`tests/test_retrieval_fusion.py`), which is what CI can run without a corpus

**#4 Forced-memo turn misses the prompt cache**: PR #105
- [x] Final forced-memo call sends `tools=TOOLS` and the structured `system` block with `cache_control`
- [x] `_roll_cache_breakpoint(messages)` before the forced-memo call and before the `max_tokens` continuation call
- [x] Continuation call also sends `tools=TOOLS`
- [x] Test: both calls carry a `cache_control` breakpoint and the tool list
- [ ] Confirm on one live run that `cache_read_ratio` in the `run_summary` line improves — **not done:** costs a real run; the mechanism is pinned by test instead

**#5 Cross-sub-query merge**: PR #104
- [x] `retrieve_full` sums each chunk's per-sub-query RRF contributions instead of taking the max
- [x] ~~Same in `retrieve_with_decomposition` if it survives #3~~ — it did not survive; deleted in #104
- [x] `gather_extraction_chunks` and `retrieve_for_extraction` run their fixed queries with `asyncio.gather`
- [ ] Measure with the #3 harness before and after — **not done:** same corpus/spend constraint as #3's re-run

**#6 Delete the dead PDF pipeline**: PR #106
- [x] Delete `app/ingest.py`, `app/retrieve.py`, `app/db_postgres.py`, `app/db.py`, `app/chunk.py`
- [x] `git rm --cached app/__init__.pyc`; add `*.pyc` to `.gitignore`
- [x] Remove `_DEAD_MODULES` from `tests/test_config.py` — the dotenv guard then covers all of `app/`
- [x] Drop the now-unused `Chunk`/`Chunks` import from `app/llm.py`
- [x] Remove the stale `.claude/worktrees/inspiring-kilby-ec9f13` worktree (`git worktree remove`)

**#7 One way to read a required setting**: PR #106
- [x] `require_env` at `main.py:410,440`, `cli.py:169,361`, `repositories/db.py:13`, `checkpointer.py:22`
- [x] Document `AGENT_TOOL_CONCURRENCY` in `.env.example`
- [x] Test: every `os.environ[...]` in `app/` is gone, in the style of `test_only_app_config_loads_dotenv`

**#8 Cost log**: PR #106
- [x] One `COST_LOG_PATH`, resolved from the repo root and overridable by env, imported by `researcher.log_cost` and `cost_log.py`
- [x] Monthly rotation; `_disk_logged_events` reads the current file plus the previous one
- [x] Test: a process started from another working directory writes to the same file

**#9 Shared structured-call helper**: PR #107
- [x] One module owning `_accumulate`, `_tool_block`, `_extract`, `_CORRECTION`, `_retry_messages`, `_maybe_crash` and the spend ceiling
- [x] debate, risk, synthesis and news ports call it; per-port constants stay per-port
- [x] The four `_assert_within_budget` variants become one, raising `NodeBudgetExceeded` as they do now
- [x] Existing port tests pass unchanged — this is a refactor, not a behaviour change

**#10 Remove the `use_hybrid` knob**: PR #104
- [x] Drop the parameter from `RetrievalService.__init__` and from `main.py:262,335`, `cli.py:402`, the test
- [x] `eval/runner.py` keeps its own mode flag under a name that says it is the harness's (`mode=`), not the service's

**#11 Failures that read as success**: PR #105
- [x] `_try_yfinance` / `_try_finnhub` log the exception (vendor, ticker, `as_of`, exception type) before returning `None`
- [x] Guard `resp.content` in `llm.py:69` and `query_decomposer.py:160`; return a clear error, not `IndexError`
- [x] `researcher.main()` sets `mode = "test"` on the `--test` branch; test it
- [x] Test: an empty-content provider response from `/ask` returns an explicit "the model returned no answer" body rather than a traceback. **Deviation:** a 200 with a plain answer, not a 4xx/5xx — the research agent treats a non-200 as a tool error and retries, which would spend a second call on a question the provider has already declined to answer

### Low
- [ ] Remove the unused imports and symbols listed under #12, including the duplicate `format_citation_tag` in `main.py` and `FinancialMetricsResponse`
- [ ] Delete `USE_STUBS`, `_stub()` and the "STEP 2" comments from `tools.py`; drop the unused `session_factory` from `MetricsRepository`; drop the unused `wall_clock_timeout_s` from `_describe_stale_budget`
- [ ] `calculate` returns its unit alongside the number; `_scale_by_value` keys on `(value, unit)` so same-valued inputs cannot collide
- [ ] Correct the `search_by_embedding` filtering docstring; evaluate `hnsw.iterative_scan` or exact search under a ticker filter
- [ ] `get_running_loop()` in `edgar/client.py`; read `filings.files` so prolific filers are not truncated
- [ ] `API_BASE` reads an env var; build one `httpx.AsyncClient` per agent run instead of one per tool call
- [ ] Add a budget check between turns inside `_sample_additional_risk_panel`
- [ ] Declare `pyyaml` and `pandas` in `pyproject.toml`
- [ ] Merge the duplicate docs: one `architecture.md`, one `trading-agent-known-gaps.md`, one `watchlist.yaml` (keep the root copy the code reads, or move the path into config and keep the `docs/` copy)
- [ ] Refresh the README: drop the fixed "budget is checked on edges" limitation, update the test counts
- [ ] Fix the `/ask` system-prompt typos in `app/llm.py`; `chmod 644 docs/tutorial.md`
- [ ] Normalise `memo_verifier`'s two separator spellings

### CI and tooling
- [ ] Add `ruff check` (and `ruff format --check`) to `.github/workflows/tests.yml`; fix what it finds
- [ ] Add a type checker in non-blocking mode first, then tighten — #1 is a type error a checker would have caught
- [ ] Commit the `docs/validation/` fixture `test_manifest_merge.py` needs, or rewrite it against a temp file, so nothing skips in CI
- [ ] Run `eval/` in CI against the committed test set once #3 lands

---

## What this pass shipped

PRs #102–#107, stacking in that order. Each is reviewable on its own top commit.

| PR | Findings | Substance |
|---|---|---|
| #102 | High #1 | `extract-metrics` called five things that do not exist; the row type is renamed and built through one constructor. A migration adds the `reasoning` column all three read methods already selected. |
| #103 | High #2 | `as_of` reaches the fundamentals leg and is enforced in the TOOLS — `filed_before` on every filing-reading call — not in the prompt. |
| #104 | Medium #3, #5, #10 | The eval harness measures `retrieve_full`; sub-queries fuse by summing; `use_hybrid` removed. |
| #105 | Medium #4, #11 | One request shape per turn, so the forced-memo call can hit the prefix cache; three silent failures now say something. |
| #106 | Medium #6, #7, #8 | Dead PDF pipeline and a Python 2 `.pyc` deleted; `require_env` everywhere; one cost-log path, rotated monthly. |
| #107 | Medium #9 | 250 lines out of three ports into one `structured_call` module. |

Two findings grew in scope once opened, and both are worth knowing about:

- **#1 was five bugs, not one.** `FilingStatus.INGESTED`, `list_by_state`, `set_state` and `Filing.fiscal_period` do not exist either. The command had never run. The repository's three read methods were also selecting a `reasoning` column the table never had.
- **#4 is worse than it reads on Anthropic.** The providers this project runs on do automatic *prefix* caching and `cache_control` is stripped in translation, so dropping `tools` from the forced-memo call did not merely lose a breakpoint — it made a cache hit impossible.

## What the previous review left open

For the record, and because these are now folded into the checklist above rather than tracked separately:

- Previous **High #1–#3** and **Medium #4–#13**: all shipped (PRs #82–#95). Spot-checked this review — CORS is scoped, `APP_API_KEY` guards the spending endpoints, ticker validation reaches the vault writer, Postgres binds to `127.0.0.1:6432`, the BM25 query ORs its terms, per-run state lives in ContextVars, `_chunk` is idempotent under a unique index, and `check_run_guards` is called from inside both expensive nodes.
- Previous **Low**: every item still open. They appear above as #6, #9, #11, #12 and the Low section.
- Previous **follow-ups left open** and still open: re-chunk and re-embed the corpus so the #94 table-aware chunking takes effect on already-ingested filings; run the synthesizer's verdict samples concurrently; give `ask_edgar` a filing-type and date filter; run `/trading/analyze` in the background behind a job id.
