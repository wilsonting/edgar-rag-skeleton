"""Wraps researcher.py's existing agent for the trading pipeline.
Deliberately calls the same path as `python -m app.agent.researcher TICKER`
(full checklist mode) — not /ask, which is a different agent behavior.
"""
import json
import os
from datetime import date
from pathlib import Path

from app.agent.researcher import AGENT_MODEL, _save_output, log_cost, run_agent
from app.agent.tools import get_delegated_usage
from app.agent.prompts import ANALYST_SYSTEM_PROMPT
from app.agent.trading.domain.fundamentals_report import FundamentalsReport
from app.agent.trading.infrastructure.cost_log import new_event_id, record_cost_event

_CACHE_DIR = Path(__file__).resolve().parents[1] / ".fundamentals_cache"
_USE_MOCK = os.getenv("MOCK_FUNDAMENTALS", "").strip() == "1"


def _cache_path(ticker: str) -> Path:
    return _CACHE_DIR / f"{ticker.upper()}.json"


async def get_fundamentals_report(
    ticker: str, run_id: str | None = None
) -> FundamentalsReport:
    cached = _cache_path(ticker)

    if _USE_MOCK and cached.exists():
        report = FundamentalsReport.model_validate_json(cached.read_text())
        age_days = (date.today() - report.generated_at).days
        print(f"[fundamentals] loading cached report for {ticker} ({age_days}d old)")
        return report

    today = date.today()
    task = f"Today's date is {today.isoformat()}. Run the full research checklist for {ticker}."
    result, usage = await run_agent(task, ANALYST_SYSTEM_PROMPT)

    event_id = new_event_id("fundamentals")
    cost = log_cost(ticker, "trading-fundamentals", usage, run_id=run_id, event_id=event_id)
    vault_path = _save_output(result, ticker.upper(), "fundamentals", cost_usd=cost)
    print(f"[fundamentals] saved memo to {vault_path}")

    # What the agent's TOOLS spent server-side, which until 2026-08-27
    # reached neither the cost log nor `check_run_guards`. Logged as its own
    # line and its own CostEvent, under a distinct mode, so the two kinds of
    # spend stay tellable apart in `docs/cost-log.jsonl`.
    tool_usage = get_delegated_usage()
    tool_event = None
    if not tool_usage.is_empty:
        tool_event_id = new_event_id("fundamentals-tools")
        tool_cost = log_cost(
            ticker, "trading-fundamentals-tools", tool_usage,
            run_id=run_id, event_id=tool_event_id,
        )
        tool_event = record_cost_event(
            tool_event_id, "fundamentals-tools", tool_usage, AGENT_MODEL, tool_cost
        )

    report = FundamentalsReport(
        ticker=ticker,
        summary=result,
        input_tokens=usage.input_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        output_tokens=usage.output_tokens,
        generated_at=today,
        cost_event=record_cost_event(event_id, "fundamentals", usage, AGENT_MODEL, cost),
        tool_cost_event=tool_event,
    )

    _CACHE_DIR.mkdir(exist_ok=True)
    cached.write_text(report.model_dump_json(indent=2))

    return report
