# Live-run check: FIG, 2026-09-13

First full-pipeline run after the code-review fixes (PRs #102–#108). Purpose was to
verify the eleven High/Medium fixes against a real run rather than against the suite.

**Command:** `python -m app.agent.trading.interface.cli FIG --thread-id trading-FIG-postreview-20260913`
**Code:** `main` at `6d7c29f` (all six fix PRs merged) · **Corpus:** warm — FIG 6 filings, 789 chunks, all embedded
**Models:** answer + extraction on `deepseek-v4-flash`, everything else on `gpt-5.6-luna`

---

## Result

| | this run | prior FIG runs |
|---|---|---|
| Cost | **$0.159018** | $0.2616 · $0.2795 · $0.8628 |
| Wall clock | **286.2s** | 595s · 613s · 224s |
| Outcome | `completed` | `completed` ×3 |
| `cache_read_ratio` | **0.7233** | 0.658 · 0.711 · 0.665 |
| `cost_ledger_gap_usd` | **$0.00** | — |
| Schema violations / retries | **0** | — |
| Errors, crashes, aborts | **0** | — |

Verdict **HOLD** (samples `hold, sell, hold`, agreement 0.67), evidence quality 0.89,
8 data gaps — all of them real guard output, not noise. The debate hit its round cap;
the risk panel ran all three verdict samples. Nothing in the pipeline degraded.

The run is cheaper and faster than every prior FIG run, but the model routing also
changed between them, so **this is not a clean attribution to the review fixes.**

---

## Two problems found

### 1. The API server was 22 hours stale — the run used pre-review server code

`uvicorn app.main:app` had been running since **Sep 12 09:06:44**, which predates every
one of the six fix PRs (#108 merged Sep 13 07:02, five minutes before the run started).
The trading CLI ran current code; the server serving its 30 `ask_edgar` calls did not.

Caught by probing `POST /latest-filings` with `filed_before`, which the server **silently
ignored** — FastAPI drops unknown request fields, so the old `LatestFilingsRequest`
accepted the request and discarded the bound:

```
before restart:  echoed filed_before=None   5 filings   max 2026-08-05   past the bound: YES
after  restart:  echoed filed_before=...    2 filings   max 2025-11-05   past the bound: no
```

**Consequence for this run:** every `/ask` went through the *old* `retrieve_full`, so the
cross-sub-query fusion fix (#5) was never exercised. The run's retrieval quality reflects
the max-merge this review replaced, not the sum-merge that shipped.

### 2. `alembic upgrade head` had not been run

The `reasoning` column from PR #102's migration was missing. Applied before the run
(`7c3e9a1f5b2d → 9f2a7c1d4e83`). Nothing in the pipeline reads it, so the run was
unaffected — but `MetricsRepository`'s three read methods would still have raised
`UndefinedColumn`.

---

## What the run actually verified

- [x] **#2 — `as_of` reaches the fundamentals leg.** `[fundamentals] running for FIG as of 2026-09-13`. The node receives the date and passes it; previously it never saw one.
- [x] **#2 — the historical-run caveat stays quiet on a current run.** `as_of == today`, and no "prior knowledge is not bounded" gap appears. The caveat logic fires on date, not unconditionally.
- [x] **#8 — cost-log rotation.** All 45 lines went to the new `docs/cost-log-2026-09.jsonl`; the 1.0 MB `cost-log.jsonl` was untouched.
- [x] **#8 — disk reconciliation still finds a run across the rotation.** `cost_ledger_gap_usd = $0.00`, `n_events = 44`, matching state exactly.
- [x] **#9 — the shared `structured_call` module.** 39 forced tool calls (6 debate turns, 27 risk turns, 3 research-manager, 3 risk-judge) across all three refactored ports. **Zero** schema violations, zero retries, zero errors.
- [x] **#6 / #7 — config and dead-code removal.** CLI and server both start and run clean with the dead modules gone and `require_env` on every required setting.
- [x] **Verdict-consistency guard (#100) is correctly silent.** Executive Summary and Assessment section both say `INSUFFICIENT_EVIDENCE`, and item 10(a) is a genuine Data Gap, which is what forces that verdict. No false flag.
- [x] **#103 — `/latest-filings` date bound**, verified against live EDGAR after the restart (above). No LLM spend.

## What the run did NOT verify

- [ ] **#4 — the forced-memo prefix-cache fix.** The agent finished naturally in **10 turns of 45**, so the forced-memo path never executed. The fundamentals loop showed 85.6% cache-read on 28,630 uncached input tokens — squarely inside the 49.5–88% / 22–32k baseline band, i.e. unchanged, because the changed code did not run. Needs a run that hits `MAX_TURNS` or the budget stop.
- [ ] **#5 — cross-sub-query RRF fusion.** Ran against the stale server (see above). Unexercised.
- [ ] **#2 — the `filed_before` bound on `ask_edgar`.** `as_of` was today, so the bound is a no-op by construction. Needs a `--as-of` in the past.
- [ ] **#3 / #10 — eval harness mode and `use_hybrid`.** Not part of a pipeline run.
- [ ] **#11 — the three silent failures.** No vendor failure, no empty completion, and `--test` is not on the pipeline path. Unit-tested only.
- [ ] **#1 — `extract-metrics`.** Not part of the pipeline; covered by `tests/test_cli_commands.py`.

---

## Fixes applied during this check

- [x] Applied the pending migration: `uv run alembic upgrade head` (`9f2a7c1d4e83`, adds `financial_metrics.reasoning`)
- [x] Restarted the API server so it serves current code
- [ ] **Add a version probe so a stale server cannot go unnoticed.** This failure was silent and cost a whole run's worth of verification. `/corpus-status` (or a `/health`) should report the running code's git SHA, and the trading CLI should warn when it differs from the working tree.
- [ ] **Make `/ask` and `/latest-filings` reject unknown request fields** (`model_config = ConfigDict(extra="forbid")`). A bound the client sends and the server silently discards is the exact failure above, and it would have surfaced as a 422 on the first call instead of a wrong answer 30 calls later.

## Observation worth acting on separately

The single largest cost in the run is **not** the agent loop:

| line | tokens in | cache read | cost | share of run |
|---|---|---|---|---|
| `trading-fundamentals` (agent loop) | 28,630 | 169,707 (85.6%) | $0.0168 | 11% |
| `trading-fundamentals-tools` (30 × `/ask`, deepseek) | **172,336** | 4,864 (**2.7%**) | **$0.0889** | **56%** |

The server-side answer calls are 56% of the run and essentially uncached — each `/ask`
sends 8 fresh chunks (~5,700 tokens) behind a ~300-token cacheable system prompt, so
prefix caching has almost nothing to bite on. #4 improved the other 11%. If run cost
becomes a target again, this is where the money is — but note `ASK_EDGAR_K = 5` was
already tried and reverted on retrieval-coverage grounds (`app/agent/tools.py`), so it
is not a free win.

---

---

# Run 2 — FIG, `--as-of 2026-03-01`, restarted server

Purpose: exercise what run 1 could not — the retrieval bound (#2) and the fusion
change (#5), both against current server code.

**Outcome: failed (exit 1)** in the technical node, after the fundamentals leg
completed. $0.070 spent. The failure is a flaky vendor, not a regression — but the way
it failed exposed two real defects, both now fixed.

## What run 2 established

### #2's `filed_before` bound is real, and measurable

| | run 1 (unbounded) | run 2 (`<= 2026-03-01`) |
|---|---|---|
| FIG chunks visible | 789 | **437** |
| distinct filing dates | 5 | **2** |
| `ask_edgar` calls | 30 (the cap) | **24** |
| delegated input tokens | 172,336 | **99,308** |
| delegated cost | $0.0889 | **$0.0550** |

Token volume tracks the corpus reduction almost exactly (58% vs 55%). The bound reached
`/ask` and changed what came back.

The fundamentals memo says so in its own words:

> **Filings reviewed:** FIG Form 10-K, filed 2026-02-18 … The corpus also contains
> filings dated after the analysis cutoff, including Form 10-Q filed 2026-05-14 and
> Form 10-Q filed 2026-08-05; **these were not used because they were unavailable as of
> 2026-03-01.**

### #5's fusion ran, against current server code

All 24 `ask_edgar` calls went through the restarted server, so `retrieve_full`'s
sum-merge was exercised. Whether it retrieves *better* is not measurable from a
pipeline run — that needs `eval/` with the `full` mode, and the corpus it needs is
gitignored.

## Two defects run 2 exposed

### A. The memo quote above is itself a lookahead leak

`ask_edgar` and `/latest-filings` were bounded; **`check_corpus` was not**. So the run
could not READ the post-cutoff filings but could still SEE them, and it enumerated two
of them by date in its own memo. Knowing a filing exists, and when, is information from
after the cutoff — the same argument that put the bound on `/latest-filings`.

Fixed: `filed_before` now flows through `/corpus-status` into all four
`CorpusStatusQuery` methods and the `check_corpus` tool. Verified live:

```
/corpus-status?ticker=FIG                          6 filings  2025-11-05 .. 2026-08-05  789 chunks
/corpus-status?ticker=FIG&filed_before=2026-03-01  2 filings  2025-11-05 .. 2026-02-18  437 chunks
```

### B. The price-vendor diagnosis was still invisible — my own #11 fix was half-done

The run died on `VendorError: No price data for FIG from yfinance or Finnhub`, with no
record of what either vendor said. Two causes:

- `#11` logged the **exception** path at WARNING but the **empty-result** path at INFO.
  yfinance returned an empty frame (transient; it returns 146 bars for that date when
  called directly), and that went to INFO.
- `app/agent/trading/interface/cli.py` **configured no logging at all** — so INFO went
  nowhere, and WARNING arrived only through Python's handler-of-last-resort, unformatted
  and unattributed. `app/cli.py` has always called `basicConfig`; the entry point that
  spends the most per invocation never did.

Fixed both. Returning `None` from a vendor helper is never routine — it either triggers
the fallback or ends the run — so it is WARNING, not INFO. Same failure now reads:

```
WARNING app...price_data_port: yfinance returned no bars for FIG as of 2026-03-01
WARNING app...price_data_port: finnhub failed for FIG as of 2026-03-01
                               FinnhubAPIException(status_code: 403): You don't have access to this resource.
VendorError: No price data for FIG from yfinance or Finnhub
```

Finnhub's free tier has no historical candles, so **historical runs depend entirely on
yfinance**, which is flaky. That is a standing constraint on `--as-of` runs, not a bug.

## Fixes applied during run 2

- [x] `check_corpus` / `/corpus-status` bounded at the analysis date (4 query methods, the endpoint, the tool)
- [x] Vendor no-data logged at WARNING, not INFO — it is never a routine outcome
- [x] `logging.basicConfig` in the trading CLI, matching `app/cli.py`
- [x] Tests for all three, including one quoting the leaked memo sentence as the reason

---

# Runs 3 and 4 — forcing the memo path, to verify #4

#4 changed the call the agent makes when its loop ends: it used to send no `tools` and a
bare-string `system`, which on these providers changes the request at position zero and
makes a prefix-cache hit impossible. Neither earlier run reached that call — run 1
finished in 10 turns of 45, run 2 died first.

**Run 3** (`LOOP_MAX_TURNS=4 --only fundamentals`) still finished naturally: with a low
cap, `TURN_WARN_AT=8` tells the agent to wrap up on every turn, so it returns prose
before the cap bites. Useful anyway — it validated the measurement method, since the
node's logged totals matched the per-turn traces exactly, residual zero:

```
node total       : in=13131  cache_read=49415  out=2938
traced 4 turns   : in=13131  cache_read=49415  out=2938
UNTRACED residual: in=0      cache_read=0      out=0
```

**Run 4** (`LOOP_MAX_TURNS=1`) reached it: `[MAX_TURNS reached — forcing memo from
gathered data]`. Subtracting the traced turn from the node's total isolates the
forced-memo call:

```
node total       : in=1685  cache_read=17873  out=1427
traced loop turn : in=110   cache_read=8919   out=63
FORCED-MEMO CALL : in=1575  cache_read=8954   out=1364
```

**8,954 of 10,529 prompt tokens (85.0%) served from cache** on the call that used to be
guaranteed to miss.

## The counterfactual, measured

Inference from one number is weak, so the two shapes were run against the provider with
**identical messages**, new shape first to populate the cache:

| call | `input` | `cache_read` | cached |
|---|---|---|---|
| 1. new shape (populates) | 5,823 | 8,847 | 60.3% |
| 2. new shape again — **the fix** | **75** | **14,595** | **99.5%** |
| 3. **old shape**, same messages | **13,217** | **0** | **0.0%** |

The old shape gets **zero** cache reads on message content the provider has just seen,
because dropping the tools block and flattening `system` to a string changes the prefix
from its first byte. Priced at this run's model, that one call is **$0.004280 old vs
$0.001944 new — 2.2× cheaper**, and it is the call carrying the entire conversation.

**#4 is verified.**

## Still open

- [x] ~~#4 unverified~~ — **verified above** (runs 3–4 plus the A/B).
- [x] Whether retrieval improved — **measured**, see below.
- [x] A technical-node vendor failure killing the run — **fixed**, see below.
- [x] Version probe and `extra="forbid"` — **both shipped**, see below.

---

# Closing the four open actions

## 1. Did retrieval actually improve? Yes — in rank, not in recall

`scripts/probe_fusion_ab.py` settles it without an LLM and without a hand-labelled gold
set, by **known-item retrieval**: a distinctive sentence lifted out of a real chunk has
that chunk as its correct answer by construction. Two such sentences make a two-part
question whose right answer set is known exactly — the shape decomposition produces, and
the only shape where the two merges can differ at all.

Two shapes, because the change could plausibly hurt one of them:

**AGREE** — two parts of *one* question (what the decomposer actually emits). 150
questions, k=8, 2.49 chunks found by both sub-queries, **102/150 rankings differ**:

| merge | recall@8 | both@8 | **MRR** |
|---|---|---|---|
| max (old) | 0.880 | 0.880 | **0.593** |
| sum (new) | 0.887 | 0.887 | **0.694** |

**SPLIT** — two *unrelated* parts with two different answers. 40 questions, **0.00**
chunks found by both sub-queries, **0/40** rankings differ — the merges are identical by
construction, because disjoint result sets make a sum a max.

**Read it honestly:** the change does not alter *whether* the answer is retrieved —
recall@8 is flat at ~0.88, and it was already high. It alters *where the answer lands*:
**MRR 0.593 → 0.694, +17%**, stable across sample sizes (+14.6% at n=40). That matters
because `ask_edgar` hands the answer model 8 chunks in order, so earlier is read better.
The feared regression on split questions does not occur.

Caveat worth keeping: these are synthetic sub-queries, not the live decomposer's output.
It measures the merge, which is what changed — not end-to-end answer quality.

## 2. A vendor outage no longer costs the run

`technical_node` let `VendorError` propagate, which ended the whole run and discarded the
fundamentals leg that had already completed and been paid for — $0.070 on run 2. Nothing
about a price feed being down invalidates the filing analysis.

It now degrades. The synthesizer already reports an absent analyst as a data gap; the new
`analyst_failures` channel makes this one say **why**, which an unselected analyst cannot
claim:

- failed: `technical analyst FAILED and this memo carries no technical evidence as a result: No price data for FIG from yfinance or Finnhub`
- unselected: `technical analyst did not run — … which is not the same as that evidence being neutral`

Same hole, different claim about it. Neither ever reads as neutral evidence. The gap
builder is `_missing_analyst_gaps`, extracted so it is testable on its own, and a
malformed diagnostic string falls back to the "did not run" wording rather than crashing
the memo.

## 3. A stale server now announces itself

`GET /health` returns the commit the **running process** started with — snapshotted at
import, never recomputed, because a value that tracks the working tree always agrees and
is exactly the comparison that needs to be able to fail. The server logs it on startup,
and the agent's tool layer probes it **once per process** (not once per `ask_edgar` call)
and warns on mismatch. A server too old to have `/health` is itself the answer, and says
so.

```
$ curl -s localhost:8000/health
{"status":"ok","commit":"3790fb9","dirty":true}
```

## 4. Unknown request fields are a 422, not a shrug

All six request models take `extra="forbid"`. Pydantic's default is to **drop** unknown
fields, which is precisely how a stale server accepted `filed_before` on
`/latest-filings` and silently discarded the bound. Verified against the live server:

```
POST /latest-filings {"ticker":"FIG","filed_befor":"2026-03-01"}   -> HTTP 422   (was 200)
```

Between them, 3 and 4 close both halves of the run-1 failure: the server could not say
what it was running, *and* it accepted a field it did not understand. Either one alone
would have left it silent.
