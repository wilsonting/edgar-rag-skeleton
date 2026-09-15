"""Persistence for extracted financial metrics.

Two types called `FinancialMetrics` used to exist: the Pydantic model the
extractor RETURNS (`app.application.extraction_service`), which carries only
the figures a model can read off a filing, and the row this module WRITES,
which also carries the identity of the filing those figures came from. This
module imported the first and then shadowed it with the second on the next
line, so the name meant different things either side of one import.

`app/cli.py` read that the way anyone would and passed the extractor's
output straight to `upsert`, whose first statement is `metrics.ticker` — a
field the extractor's model does not have. The row type is now
`FinancialMetricsRow`, and `from_extraction` is the one place that turns the
extractor's output into it, so the fields only this layer knows (ticker,
period, filing, citations) cannot be forgotten by a caller again.
"""

from dataclasses import dataclass
from datetime import date, datetime
import json
from typing import Literal

from app.application.extraction_service import FinancialMetrics

from app.infrastructure.repositories.db import get_connection


@dataclass(frozen=True)
class FinancialMetricsRow:
    """One `financial_metrics` row: the extracted figures, plus the filing
    identity and citations that say where they came from."""
    ticker: str
    fiscal_period: str                          # "Q1 2026"
    filing_type: str                            # "10-Q", "10-K"
    filed_date: date
    revenue: float | None                       # $M
    gross_margin_pct: float | None              # e.g. 79.0
    gaap_net_income: float | None               # $M
    free_cash_flow: float | None                # $M
    sbc_pct_of_revenue: float | None            # e.g. 50.7
    net_dollar_retention: float | None          # e.g. 139.0
    extraction_confidence: Literal["stated", "computed", "not_disclosed"]
    reasoning: str                              # model shows its work
    source_citations: list[str]
    extracted_at: datetime | None = None

    @classmethod
    def from_extraction(
        cls,
        extracted: FinancialMetrics,
        *,
        ticker: str,
        fiscal_period: str,
        filing_type: str,
        filed_date: date,
        source_citations: list[str],
    ) -> "FinancialMetricsRow":
        """The extractor's output plus the identity it cannot know.

        Used by both writers — `POST /extract` and the `extract-metrics` CLI
        command — so neither can drift into passing the extractor's model
        where a row belongs.
        """
        return cls(
            ticker=ticker,
            fiscal_period=fiscal_period,
            filing_type=filing_type,
            filed_date=filed_date,
            revenue=extracted.revenue,
            gross_margin_pct=extracted.gross_margin_pct,
            gaap_net_income=extracted.gaap_net_income,
            free_cash_flow=extracted.free_cash_flow,
            sbc_pct_of_revenue=extracted.sbc_pct_of_revenue,
            net_dollar_retention=extracted.net_dollar_retention,
            extraction_confidence=extracted.extraction_confidence,
            reasoning=extracted.reasoning,
            source_citations=source_citations,
        )


class MetricsRepository:
    async def upsert(self, metrics: FinancialMetricsRow) -> None:
        """
        Insert or update one metrics row.
        ON CONFLICT on (ticker, filing_type, fiscal_period) — re-running
        extraction on the same filing refreshes the row, never duplicates.
        """
        if not isinstance(metrics, FinancialMetricsRow):
            # The exact mistake this module's docstring describes. Named here
            # rather than left to surface as `'FinancialMetrics' object has
            # no attribute 'ticker'` several frames from the call site.
            raise TypeError(
                f"upsert expects a FinancialMetricsRow, got "
                f"{type(metrics).__name__}. Build one with "
                f"FinancialMetricsRow.from_extraction(...), which supplies "
                f"the ticker, fiscal period, filing and citations the "
                f"extractor's own model does not carry."
            )
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO financial_metrics (
                        ticker, fiscal_period, filing_type, filed_date,
                        revenue, gross_margin_pct, gaap_net_income,
                        free_cash_flow, sbc_pct_of_revenue, net_dollar_retention,
                        extraction_confidence, reasoning, source_citations
                    )
                    VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s
                    )
                    ON CONFLICT ON CONSTRAINT uq_metric_period DO UPDATE SET
                        revenue                = EXCLUDED.revenue,
                        gross_margin_pct       = EXCLUDED.gross_margin_pct,
                        gaap_net_income        = EXCLUDED.gaap_net_income,
                        free_cash_flow         = EXCLUDED.free_cash_flow,
                        sbc_pct_of_revenue     = EXCLUDED.sbc_pct_of_revenue,
                        net_dollar_retention   = EXCLUDED.net_dollar_retention,
                        extraction_confidence  = EXCLUDED.extraction_confidence,
                        reasoning              = EXCLUDED.reasoning,
                        source_citations       = EXCLUDED.source_citations,
                        extracted_at           = now()
                    """,
                    (
                        metrics.ticker.upper(),
                        metrics.fiscal_period,
                        metrics.filing_type,
                        metrics.filed_date,
                        metrics.revenue,
                        metrics.gross_margin_pct,
                        metrics.gaap_net_income,
                        metrics.free_cash_flow,
                        metrics.sbc_pct_of_revenue,
                        metrics.net_dollar_retention,
                        metrics.extraction_confidence,
                        metrics.reasoning,
                        json.dumps(metrics.source_citations),
                    ),
                )
                await conn.commit()

    async def get(self, ticker: str, fiscal_period: str) -> FinancialMetricsRow | None:
        """Fetch one row by ticker + period. Returns None if not yet extracted."""
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT {_COLUMNS} FROM financial_metrics "
                    f"WHERE ticker = %s AND fiscal_period = %s",
                    (ticker.upper(), fiscal_period),
                )
                row = await cur.fetchone()

        if row is None:
            return None
        return self._row_to_metrics(row)

    async def list_by_ticker(self, ticker: str) -> list[FinancialMetricsRow]:
        """All periods for one ticker, oldest first — this is your trend query."""
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT {_COLUMNS} FROM financial_metrics "
                    f"WHERE ticker = %s ORDER BY filed_date ASC",
                    (ticker.upper(),),
                )
                rows = await cur.fetchall()

        return [self._row_to_metrics(r) for r in rows]

    async def list_by_tickers(self, tickers: list[str]) -> list[FinancialMetricsRow]:
        """
        Cross-ticker comparison — same columns, multiple tickers.
        e.g. FIG vs ADBE gross_margin_pct side by side.
        """
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT {_COLUMNS} FROM financial_metrics "
                    f"WHERE ticker = ANY(%s) ORDER BY ticker ASC, filed_date ASC",
                    ([t.upper() for t in tickers],),
                )
                rows = await cur.fetchall()

        return [self._row_to_metrics(r) for r in rows]

    @staticmethod
    def _row_to_metrics(row: dict) -> FinancialMetricsRow:
        """Shared row → dataclass conversion. Mirror of Chunk.model_validate pattern."""
        citations = row["source_citations"]
        return FinancialMetricsRow(
            ticker=row["ticker"],
            fiscal_period=row["fiscal_period"],
            filing_type=row["filing_type"],
            filed_date=row["filed_date"],
            revenue=row["revenue"],
            gross_margin_pct=row["gross_margin_pct"],
            gaap_net_income=row["gaap_net_income"],
            free_cash_flow=row["free_cash_flow"],
            sbc_pct_of_revenue=row["sbc_pct_of_revenue"],
            net_dollar_retention=row["net_dollar_retention"],
            extraction_confidence=row["extraction_confidence"],
            reasoning=row["reasoning"] or "",
            source_citations=json.loads(citations) if isinstance(citations, str) else citations,
            extracted_at=row["extracted_at"],
        )


# One list, used by every read. `reasoning` is in it, and until the migration
# beside this change the column did not exist — so all three read methods
# raised UndefinedColumn. Nothing called them, which is why that sat unseen.
_COLUMNS = """
    ticker, fiscal_period, filing_type, filed_date,
    revenue, gross_margin_pct, gaap_net_income,
    free_cash_flow, sbc_pct_of_revenue, net_dollar_retention,
    extraction_confidence, reasoning, source_citations, extracted_at
"""
