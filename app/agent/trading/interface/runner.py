"""Start, resume or replay one trading run — shared by the CLI and the API.

Exists because the two entry points had drifted apart. The CLI seeded the
graph with `as_of_date`, `run_id` and a `RunBudget`, set a recursion limit,
refused a resume whose deadline had passed, logged a run summary and saved
the vault artifacts. `POST /trading/analyze` did none of that: it invoked
the graph with `{"ticker": ...}` alone, so the fundamentals node ran (the
most expensive one) and the technical node then raised on the missing
`as_of_date` — a 500 after the money was spent, with no budget guard,
because `_guarded()` reads a missing budget as "opted out". Everything a
run needs before its first node now lives here, so an entry point cannot
leave part of it out.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Literal

from app.agent.trading.application.debate_router import MAX_ROUNDS
from app.agent.trading.application.risk_router import RISK_MAX_ROUNDS
from app.agent.trading.domain.budget import RunBudget, RunTermination
from app.agent.trading.infrastructure.cost_log import log_run_summary
from app.agent.trading.infrastructure.debate_port import save_debate_transcript
from app.agent.trading.infrastructure.decision_memo_port import save_decision_memo
from app.agent.trading.infrastructure.graph import ANALYST_CHAINS
from app.agent.trading.infrastructure.news_digest_port import save_sentiment_report
from app.agent.trading.infrastructure.risk_port import save_risk_transcript

# Gate B (Phase 6 plan §0): recursion_limit is a GLOBAL super-step budget for
# the whole invocation, not per-cycle — Phase 5's "2 * MAX_ROUNDS + 12" only
# covered one cycle plus a fixed 12 for "other nodes + slack". Phase 6 adds a
# second cycle of comparable depth (three personas instead of two sides), so
# reusing that literal would trip a legitimate run mid-risk-round with a
# GraphRecursionError that reads like a hung risk panel rather than what it
# actually is: an under-sized global counter. Every term here is derived,
# never a literal, for the same reason Phase 5's version was: a hardcoded
# number silently becomes wrong the day MAX_ROUNDS or RISK_MAX_ROUNDS moves.
_FIXED_NODES = sum(len(chain) for chain in ANALYST_CHAINS.values())  # worst case: all analysts selected
_FIXED_NODES += 4   # debate_close, risk_close, synthesizer, graceful_abort (Phase 8)
RECURSION_LIMIT = (
    2 * MAX_ROUNDS            # debate turns (bull/bear alternation)
    + 3 * RISK_MAX_ROUNDS     # risk turns (three-persona rotation)
    + _FIXED_NODES
    + 5                       # headroom, matching Phase 5's margin
)

# Phase 7 battery measured $2.24 over 5 tickers (~$0.448/run) BEFORE prompt
# caching was measured; the cache_control breakpoints already in debate_port/
# risk_port/synthesis_port were already live at that point but never
# separately verified, so 0.60/0.75 are the pre-caching-informed target/hard
# cap, to be recalibrated once a real run's cache_read_ratio is measured
# (docs/cost-log.jsonl run_summary lines, criterion 3).
DEFAULT_MAX_USD = 0.75
# No prior measurement of real end-to-end wall-clock time exists — generous
# on purpose, since the deadline exists to catch a genuine hang, not to race
# a normal run. Override with --wall-clock-timeout-s for the breach test.
DEFAULT_WALL_CLOCK_TIMEOUT_S = 1800


RunStatus = Literal["completed", "resumed", "started", "refused"]


@dataclass
class RunOutcome:
    """What `start_or_resume` did.

    `result` is the graph state after the call — the replayed state for an
    already-completed thread — and None only when a resume was refused, in
    which case `refusal` says why.
    """

    status: RunStatus
    thread_id: str
    result: dict | None = None
    refusal: str | None = None


def default_thread_id(ticker: str, analysts: list[str] | None = None) -> str:
    """A subset run has a different topology, so it gets its own default
    thread: resuming a full run's checkpoint under a narrower graph would
    report the cached fundamentals/technical of an earlier run as if this
    run produced them. An explicit thread id still overrides, deliberately."""
    suffix = "" if analysts is None else "-" + "+".join(sorted(analysts))
    return f"trading-{ticker}{suffix}"


def initial_state(
    ticker: str,
    as_of: date,
    run_id: str,
    max_usd: float = DEFAULT_MAX_USD,
    wall_clock_timeout_s: float = DEFAULT_WALL_CLOCK_TIMEOUT_S,
    now: datetime | None = None,
) -> dict:
    """The state a NEW run starts from. `as_of_date`, `run_id` and `budget`
    are set once here, at the boundary, and never recomputed inside a node —
    a resumed run reuses whatever its checkpoint already holds, and never
    moves `deadline_utc` relative to the resume time."""
    start = now or datetime.now(timezone.utc)
    return {
        "ticker": ticker,
        "as_of_date": as_of,
        "run_id": run_id,
        "budget": RunBudget(
            max_usd=max_usd,
            deadline_utc=start + timedelta(seconds=wall_clock_timeout_s),
        ),
    }


def _describe_stale_budget(values: dict, max_usd: float, wall_clock_timeout_s: float) -> str | None:
    """Refuse a resume whose inherited deadline has already passed, and say
    so when the inherited budget disagrees with the flags just given.

    Returns an error string to print, or None to proceed. Checked BEFORE
    `ainvoke` so a doomed resume costs nothing: the run-level guards can only
    fire between nodes, which on this graph means after the fundamentals
    stage has already been paid for.
    """
    budget = values.get("budget")
    if budget is None:
        return None

    now = datetime.now(timezone.utc)
    lines = []
    if budget.deadline_utc <= now:
        overdue = now - budget.deadline_utc
        lines.append(
            f"REFUSING TO RESUME: this thread's deadline passed "
            f"{_humanize(overdue)} ago ({budget.deadline_utc.isoformat()}).\n"
            f"The deadline is an absolute instant fixed when the run first "
            f"started, not a fresh window per attempt, so resuming would run "
            f"the expensive analyst stages and then abort on the first guard "
            f"check — paying full price for no memo."
        )
    if abs(budget.max_usd - max_usd) > 1e-9:
        lines.append(
            f"NOTE: --max-usd {max_usd:.2f} is IGNORED on a resume; this "
            f"thread carries ${budget.max_usd:.2f} from its first attempt."
        )
    if not lines:
        return None

    lines.append(
        "Start a fresh thread instead (--thread-id ...-r2), which takes the "
        "budget and deadline from this command line."
    )
    return "\n".join(lines)


def _humanize(delta: timedelta) -> str:
    hours, rem = divmod(int(delta.total_seconds()), 3600)
    return f"{hours}h{rem // 60:02d}m" if hours else f"{rem // 60}m"


async def start_or_resume(
    graph,
    ticker: str,
    thread_id: str,
    as_of: date,
    max_usd: float = DEFAULT_MAX_USD,
    wall_clock_timeout_s: float = DEFAULT_WALL_CLOCK_TIMEOUT_S,
) -> RunOutcome:
    """Replay a completed thread, resume an unfinished one, or start a new
    one — the three cases every entry point has to tell apart.

    A resume inherits the checkpoint's budget and deadline, never the
    arguments given here. It is the right rule and it had a hole: nothing
    said so out loud, and nothing checked whether the inherited deadline was
    already in the past. Live cost of that hole (MSFT, 2026-08-28): a thread
    whose first attempt died ~17 hours earlier was resumed with --max-usd
    1.40. The run silently used the checkpointed 1.10, executed the whole
    fundamentals stage, then aborted `deadline_exceeded` on the first guard
    check after it — $0.4069 spent, no memo.
    """
    wall_clock_start = time.monotonic()
    config = {
        "configurable": {"thread_id": thread_id},
        # Layer 2 of both cycles' termination guarantee, behind each router's
        # own cap. See RECURSION_LIMIT above for the derivation.
        "recursion_limit": RECURSION_LIMIT,
    }
    state = await graph.aget_state(config)

    if state.values and not state.next:
        return RunOutcome("completed", thread_id, result=state.values)

    if state.next:
        stale = _describe_stale_budget(state.values, max_usd, wall_clock_timeout_s)
        if stale:
            return RunOutcome("refused", thread_id, refusal=stale)
        result = await graph.ainvoke(None, config=config)
        # A resume only happens after an attempt that died mid-run — one that
        # finished or aborted gracefully reached END and replays instead — so
        # no summary exists for this run yet. Without this line a
        # crash-and-resume, the case the disk reconciliation in
        # log_run_summary exists for, left no run_summary at all.
        _log_summary(result, thread_id, ticker, as_of, wall_clock_start, resumed=True)
        return RunOutcome("resumed", thread_id, result=result)

    result = await graph.ainvoke(
        initial_state(ticker, as_of, thread_id, max_usd, wall_clock_timeout_s),
        config=config,
    )
    _log_summary(result, thread_id, ticker, as_of, wall_clock_start, resumed=False)
    return RunOutcome("started", thread_id, result=result)


def _log_summary(
    result: dict, thread_id: str, ticker: str, as_of: date,
    wall_clock_start: float, *, resumed: bool,
) -> None:
    budget = result.get("budget")
    if budget is None:
        # Only a checkpoint from before budgets existed lacks one; there is
        # no cap to report against.
        return
    log_run_summary(
        run_id=result.get("run_id") or thread_id,
        ticker=ticker,
        # A resume keeps its checkpoint's analysis date, whatever was asked.
        as_of_date=result.get("as_of_date") or as_of,
        events=result.get("cost_events") or [],
        budget=budget,
        terminated_by=result.get("run_terminated_by") or RunTermination.COMPLETED,
        wall_clock_s=time.monotonic() - wall_clock_start,
        resumed=resumed,
    )


def save_vault_artifacts(result: dict, run_log: str) -> list:
    """Write the run's artifacts once the terminal log is complete.

    Saved here rather than inside the nodes for two reasons: the log is only
    whole at the end of the run, and a resumed or already-completed run
    replays state without executing any node, which would otherwise write no
    artifact at all for a run the user just asked for.
    """
    saved = []
    digest = result.get("news_digest")
    sentiment = result.get("sentiment_summary")
    has_sentiment = digest is not None and sentiment is not None

    if has_sentiment:
        saved.append(
            save_sentiment_report(
                digest,
                sentiment,
                issues=result.get("news_digest_issues") or [],
                provenance=run_log,
            )
        )

    turns = result.get("debate_turns") or []
    if turns:
        saved.append(
            save_debate_transcript(
                result["ticker"],
                turns,
                result.get("debate_terminated_by") or "",
            )
        )

    risk_turns = result.get("risk_turns") or []
    if risk_turns:
        saved.append(
            save_risk_transcript(
                result["ticker"],
                risk_turns,
                result.get("risk_terminated_by") or "",
            )
        )

    memo = result.get("decision_memo")
    if memo is not None:
        # The log is written exactly once per run. It rides with the
        # sentiment report when there is one, and falls back to the memo
        # otherwise (e.g. `--only technical`) so a run never loses its trace.
        # Both land in the same dated folder, so the log is beside either.
        saved.append(
            save_decision_memo(memo, provenance=None if has_sentiment else run_log)
        )
    return saved
