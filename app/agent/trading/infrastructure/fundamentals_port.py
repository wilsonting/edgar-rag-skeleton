"""Wraps researcher.py's existing agent for the trading pipeline.
Deliberately calls the same path as `python -m app.agent.researcher TICKER`
(full checklist mode) — not /ask, which is a different agent behavior.
"""
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

from app.agent.researcher import (
    AGENT_MODEL,
    StopCheck,
    UsageSummary,
    _compute_cost,
    _save_output,
    log_cost,
    run_agent,
)
from app.agent.tools import get_delegated_usage
from app.agent.prompts import ANALYST_SYSTEM_PROMPT
from app.agent.trading.application.guards import check_run_guards
from app.agent.trading.domain.budget import CostEvent, RunBudget
from app.agent.trading.domain.fundamentals_report import FundamentalsReport
from app.agent.trading.infrastructure.cost_log import new_event_id, record_cost_event

_CACHE_DIR = Path(__file__).resolve().parents[1] / ".fundamentals_cache"
_USE_MOCK = os.getenv("MOCK_FUNDAMENTALS", "").strip() == "1"


def _cache_path(ticker: str, as_of: date) -> Path:
    """Keyed on the analysis date as well as the ticker.

    On ticker alone, a cached August memo answered a March run — the cache
    is written on every real run and read under MOCK_FUNDAMENTALS, so one
    environment variable was all that stood between a historical probe and
    a report from a different date.
    """
    return _CACHE_DIR / f"{ticker.upper()}-{as_of.isoformat()}.json"


def _spend_so_far(usage: UsageSummary) -> float:
    """The agent loop's own spend plus what its tools spent server-side so
    far, each priced at the model that spent it (a server that did not name
    its model is priced at the agent's, as the final accounting does)."""
    total = _compute_cost(usage, AGENT_MODEL) or 0.0
    for model, tool_usage in get_delegated_usage().items():
        total += _compute_cost(tool_usage, model or AGENT_MODEL) or 0.0
    return total


def budget_stop_check(
    budget: RunBudget | None, prior_events: list[CostEvent] | None = None
) -> StopCheck | None:
    """A `run_agent` stop check that applies the run's budget and deadline
    INSIDE the fundamentals node.

    The graph checks `check_run_guards` only on its edges, and this node is
    one edge's worth of work: up to LOOP_MAX_TURNS model calls plus up to 30
    ask_edgar/extract_metrics calls, each with its own server-side LLM
    calls. Built on `check_run_guards` itself, over the run's earlier events
    plus one in-flight event for this loop, so a breach here means exactly
    what a breach on an edge means. None when the run has no budget (tests
    and the standalone research CLI), which keeps the loop unguarded as
    before.
    """
    if budget is None:
        return None
    prior = list(prior_events or [])

    def check(usage: UsageSummary) -> str | None:
        spent = _spend_so_far(usage)
        in_flight = CostEvent(
            event_id="fundamentals:in-flight",
            node="fundamentals",
            model=AGENT_MODEL,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=usage.cache_write_tokens,
            cache_read_input_tokens=usage.cache_read_tokens,
            usd=spent,
        )
        termination = check_run_guards([*prior, in_flight], budget, datetime.now(timezone.utc))
        if termination is None:
            return None
        return (
            f"{termination.value}: ~${spent:.4f} spent in this node against a "
            f"${budget.max_usd:.2f} run budget, deadline {budget.deadline_utc.isoformat()}"
        )

    return check


async def get_fundamentals_report(
    ticker: str,
    as_of: date,
    run_id: str | None = None,
    *,
    budget: RunBudget | None = None,
    prior_events: list[CostEvent] | None = None,
) -> FundamentalsReport:
    """The fundamentals leg, bounded at `as_of` like every other source.

    `as_of` is required, not defaulted. This node used to call
    `date.today()` and never receive the run's analysis date at all, so a
    `--as-of 2026-03-01` run bounded its prices and news at March and let
    its most heavily-weighted analyst read whatever had been filed since —
    and the memo said nothing about it. Prices and news already refuse to
    run without the date (nodes.technical_node, nodes.news_node); this now
    does too, by taking it as a positional argument nothing can omit.

    The bound is enforced in the tools, not in the prompt: `run_agent`
    puts it in the run state and every filing-reading tool sends
    `filed_before` with it. Wording alone would leave the hole open on any
    turn the model did not think about it.
    """
    cached = _cache_path(ticker, as_of)

    if _USE_MOCK and cached.exists():
        report = FundamentalsReport.model_validate_json(cached.read_text())
        print(f"[fundamentals] loading cached report for {ticker} as of {as_of}")
        return report

    task = (
        f"The analysis date is {as_of.isoformat()}. Run the full research "
        f"checklist for {ticker}. Filing retrieval is bounded at that date, "
        f"so anything filed after it is deliberately unavailable to you — "
        f"report what is missing as a data gap rather than reasoning from "
        f"memory about it."
    )
    result, usage = await run_agent(
        task, ANALYST_SYSTEM_PROMPT,
        stop_check=budget_stop_check(budget, prior_events),
        as_of=as_of,
    )

    event_id = new_event_id("fundamentals")
    cost = log_cost(ticker, "trading-fundamentals", usage, run_id=run_id, event_id=event_id)
    vault_path = _save_output(result, ticker.upper(), "fundamentals", cost_usd=cost)
    print(f"[fundamentals] saved memo to {vault_path}")

    # What the agent's TOOLS spent server-side, which until 2026-08-27
    # reached neither the cost log nor `check_run_guards`. Logged as its own
    # line and its own CostEvent, under a distinct mode, so the two kinds of
    # spend stay tellable apart in `docs/cost-log.jsonl`.
    #
    # One event per model that spent it, each priced at its own rate. The
    # server's models need not be the agent's, and pricing a deepseek-served
    # /ask at a gpt-5.6-luna agent's rate under-counted it by ~44% on
    # 2026-09-11 — the budget guard waved through the difference.
    tool_events = []
    for model, tool_usage in get_delegated_usage().items():
        if tool_usage.is_empty:
            continue
        if model is None:
            print(
                f"[fundamentals] server reported tool usage without a model — "
                f"pricing it at {AGENT_MODEL}, which is wrong if the server's "
                f"answer/extraction models differ. Restart the API server."
            )
            model = AGENT_MODEL
        tool_event_id = new_event_id("fundamentals-tools")
        tool_cost = log_cost(
            ticker, "trading-fundamentals-tools", tool_usage, model,
            run_id=run_id, event_id=tool_event_id,
        )
        tool_events.append(record_cost_event(
            tool_event_id, "fundamentals-tools", tool_usage, model, tool_cost
        ))

    report = FundamentalsReport(
        ticker=ticker,
        summary=result,
        input_tokens=usage.input_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        output_tokens=usage.output_tokens,
        # The analysis date, not the wall clock: a report generated today
        # about March is a March report, and dating it today is how a
        # historical probe comes to look current.
        generated_at=as_of,
        cost_event=record_cost_event(event_id, "fundamentals", usage, AGENT_MODEL, cost),
        tool_cost_events=tool_events,
    )

    _CACHE_DIR.mkdir(exist_ok=True)
    cached.write_text(report.model_dump_json(indent=2))

    return report
