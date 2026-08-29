from __future__ import annotations

from datetime import date
from pydantic import BaseModel, Field

from app.agent.trading.domain.budget import CostEvent


class TechnicalIndicators(BaseModel):
    """Raw, Python-computed values. No LLM ever writes to this model."""
    sma_50: float | None = None
    sma_200: float | None = None
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bb_upper: float | None = None
    bb_mid: float | None = None
    bb_lower: float | None = None
    last_close: float
    volume_vs_20d_avg: float | None = None   # ratio, e.g. 1.4 = 40% above avg


class TechnicalReport(BaseModel):
    ticker: str
    as_of_date: date
    data_source: str = Field(description="'yfinance' or 'finnhub'")
    bars_used: int = Field(description="count of daily bars the calc was run on")
    bars_dropped_invalid: int = Field(
        default=0,
        description="bars the vendor returned with a missing Close or Volume, "
                    "dropped before computing. Non-zero means the vendor's "
                    "history had holes — worth knowing, because a single bad "
                    "bar can silently void an EWM-based indicator like MACD.",
    )
    indicators: TechnicalIndicators
    interpretation: str
    interpretation_flagged_claims: list[str] = Field(
        default_factory=list,
        description="statements in `interpretation` that contradict a relation "
                    "computed from `indicators` (e.g. calling price below a "
                    "moving average it is above). Distinct from "
                    "`interpretation_flagged_numbers`: those are numbers with "
                    "no source, these are claims that are false about numbers "
                    "that are real.",
    )
    interpretation_flagged_numbers: list[str] = Field(
        default_factory=list,
        description="numbers in `interpretation` that could not be matched "
                     "back to `indicators` — populated by the guard in "
                     "technical_interpreter_port.py",
    )
    cost_event: CostEvent | None = None
