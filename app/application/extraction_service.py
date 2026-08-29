from app.domain.token_usage import TokenUsage
import os
from datetime import date
from typing import Literal

from app.infrastructure.llm import get_client
from app.infrastructure.llm.models import model_for
from pydantic import BaseModel, field_validator

from app.infrastructure.repositories.chunk_repo import RetrievedChunk


class FinancialMetrics(BaseModel):
    revenue: float | None
    gross_margin_pct: float | None
    gaap_net_income: float | None
    free_cash_flow: float | None
    sbc_pct_of_revenue: float | None
    net_dollar_retention: float | None
    extraction_confidence: Literal["stated", "computed", "not_disclosed"]
    reasoning: str = ""  # forces the model to show its work — cheap sanity check

    @field_validator(
        "revenue", "gross_margin_pct", "gaap_net_income",
        "free_cash_flow", "sbc_pct_of_revenue", "net_dollar_retention",
        mode="before",
    )
    
    @classmethod
    def coerce_non_numeric_to_none(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return None
        return v

class MetricsExtractor:
    def __init__(self):
        self.llm_model = model_for("extraction")
        self.client = get_client(self.llm_model)
        # Set by every `extract` call. Starts empty so a caller that reads it
        # before any extraction gets a zero rather than an AttributeError.
        self.last_usage = TokenUsage()

    async def extract(
        self,
        chunks: list[RetrievedChunk],
        ticker: str,
        period: str,
        filing_type: str,
        filed_date: date) -> FinancialMetrics:
        context = "\n\n".join(f"[{c.chunk.section_path}]\n{c.chunk.content}" for c in chunks)
        response = await self.client.messages.create(
            model=self.llm_model,
            max_tokens=1024,
            tools=[{
                "name": "record_metrics",
                "input_schema": ExtractedFigures.model_json_schema(),
            }],
            tool_choice={"type": "tool", "name": "record_metrics"},
            messages=[{
                "role": "user",
                "content": f"""Extract financial metrics for {ticker} {period} ({filing_type}, filed {filed_date.isoformat()}) from these filing excerpts.

                Rules:
                - If a number is stated directly, confidence = "stated"
                - If you must combine two disclosed numbers to compute it (e.g. margin from revenue and cost), confidence = "computed" and show the calculation in reasoning
                - If not present in the excerpts, return null and confidence = "not_disclosed" — never estimate from general knowledge
                - Never mix time periods when computing a metric
                - Return all dollar amounts in millions (e.g., $333,439 thousand = 333.4 million)
                - Never derive a figure from a rounded narrative amount. "$1.4 billion" in prose is not a usable input; use only the exact figures in the financial statements and notes.
                - When a component breakout sums to a total, use that sum. Do not substitute a rounded figure from elsewhere in the filing.
                - Show the exact values you divided in `reasoning`.
                - Report gross profit and total stock-based compensation expense as dollar amounts in millions. Do not compute margins or percentages — those are calculated downstream.
                Excerpts:
                {context}"""
                            }])
        tool_call = next(b for b in response.content if b.type == "tool_use")
        figures = ExtractedFigures.model_validate(tool_call.input)

        # Stashed on the instance rather than added to the return type:
        # FinancialMetrics is a domain model that gets persisted and
        # compared, and token counts are neither a financial metric nor
        # something that should end up in the metrics table. The caller
        # reads `last_usage` immediately after awaiting `extract`.
        self.last_usage = TokenUsage.from_response(response.usage)

        return FinancialMetrics(
            revenue=figures.revenue,
            gross_margin_pct=_ratio(figures.gross_profit, figures.revenue),
            gaap_net_income=figures.gaap_net_income,
            free_cash_flow=figures.free_cash_flow,
            sbc_pct_of_revenue=_ratio(figures.sbc_expense, figures.revenue),
            net_dollar_retention=figures.net_dollar_retention,
            extraction_confidence=figures.extraction_confidence,
            reasoning=figures.reasoning,
        )

class ExtractedFigures(BaseModel):
    """What the LLM fills in. Figures only — no ratios."""
    revenue: float | None
    gross_profit: float | None
    gaap_net_income: float | None
    free_cash_flow: float | None
    sbc_expense: float | None
    net_dollar_retention: float | None
    extraction_confidence: Literal["stated", "computed", "not_disclosed"]
    reasoning: str = ""

    @field_validator("revenue", "gross_profit", "gaap_net_income",
                     "free_cash_flow", "sbc_expense", "net_dollar_retention",
                     mode="before")
    @classmethod
    def coerce_non_numeric_to_none(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return None
        return v


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return round(numerator / denominator * 100, 1)