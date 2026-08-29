from datetime import date
from pydantic import BaseModel

from app.agent.trading.domain.budget import CostEvent


class FundamentalsReport(BaseModel):
    ticker: str
    summary: str
    input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int
    generated_at: date
    # None on a cache hit (get_fundamentals_report returned a cached report
    # without calling the LLM) — a cached run spent nothing this run, and
    # cost_events must reflect that, not the cost of whichever run first
    # produced the cache.
    cost_event: CostEvent | None = None
    # The research agent's TOOLS spend money too: `ask_edgar` and
    # `extract_metrics` are HTTP calls to the FastAPI app that run their own
    # Claude calls server-side. Kept as a SECOND event rather than folded
    # into `cost_event` so the agent's own loop stays separately measurable
    # -- `cache_read_ratio` and the turn-count analysis both read the loop's
    # numbers, and mixing ~110k un-cached tool tokens into them would make
    # every caching metric meaningless. None when nothing was delegated.
    tool_cost_event: CostEvent | None = None