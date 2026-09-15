"""The CLI commands run at all.

`extract-metrics` shipped broken and stayed broken because nothing in the
suite invoked any `app.cli` command. It called five things that do not
exist:

  - `FilingStatus.INGESTED` — the enum has no such member
  - `filing_repo.list_by_state(...)` — the method is `list_by_ticker_and_status`
  - `filing_repo.set_state(...)` — the method is `mark_status`
  - `filing.fiscal_period` — Filing has `period_of_report`
  - `metrics_repo.upsert(extracted)` — upsert needs the ROW type, and the
    extractor returns the figures-only model, so the first statement in
    upsert (`metrics.ticker`) raised AttributeError

The last one is the one this file exists to pin: two types named
`FinancialMetrics`, one imported into the module that shadows it.
"""

from __future__ import annotations

from datetime import date

import pytest
from typer.testing import CliRunner

import app.cli as cli
from app.application.extraction_service import FinancialMetrics
from app.domain.filing import Filing
from app.domain.values import FilingStatus
from app.infrastructure.repositories.metrics_repo import (
    FinancialMetricsRow,
    MetricsRepository,
)


def _extracted(**over) -> FinancialMetrics:
    return FinancialMetrics(**{
        "revenue": 64.9, "gross_margin_pct": 32.5, "gaap_net_income": 7.3,
        "free_cash_flow": 9.0, "sbc_pct_of_revenue": 4.2,
        "net_dollar_retention": None, "extraction_confidence": "stated",
        "reasoning": "read off the consolidated statements", **over,
    })


def _filing(**over) -> Filing:
    return Filing(**{
        "id": 1, "security_id": 7, "filing_type": "10-K",
        "filed_date": date(2025, 10, 9),
        "period_of_report": date(2025, 8, 31),
        "accession_number": "0000123456-25-000001",
        "status": FilingStatus.EMBEDDED, **over,
    })


# ---------------------------------------------------------------------------
# The row type
# ---------------------------------------------------------------------------

def test_from_extraction_carries_both_halves():
    """The figures come from the extractor; the identity cannot."""
    row = FinancialMetricsRow.from_extraction(
        _extracted(), ticker="acn", fiscal_period="FY2025", filing_type="10-K",
        filed_date=date(2025, 10, 9), source_citations=["[ACN 10-K 2025 §Item 8]"],
    )
    assert row.revenue == 64.9
    assert row.reasoning == "read off the consolidated statements"
    assert (row.ticker, row.fiscal_period, row.filing_type) == ("acn", "FY2025", "10-K")
    assert row.source_citations == ["[ACN 10-K 2025 §Item 8]"]


def test_upsert_names_the_mistake_instead_of_raising_attributeerror():
    """The exact call `app/cli.py` used to make."""
    import asyncio
    with pytest.raises(TypeError) as exc:
        asyncio.run(MetricsRepository().upsert(_extracted()))
    message = str(exc.value)
    assert "FinancialMetricsRow" in message
    assert "from_extraction" in message


# ---------------------------------------------------------------------------
# The period label
# ---------------------------------------------------------------------------

def test_an_annual_report_is_labelled_by_the_filers_own_fiscal_year():
    """ACN's year ends in August; the period ending 2025-08-31 is FY2025."""
    assert _filing().fiscal_period_label() == "FY2025"
    assert _filing(filing_type="20-F").fiscal_period_label() == "FY2025"


def test_a_quarterly_report_is_labelled_by_its_period_end_not_a_guessed_quarter():
    """A calendar quarter is not a fiscal quarter, and Filing cannot tell."""
    label = _filing(filing_type="10-Q", period_of_report=date(2026, 2, 28)).fiscal_period_label()
    assert label == "period ending 2026-02-28"
    assert "Q1" not in label and "Q2" not in label


def test_a_filing_with_no_period_end_falls_back_to_the_filed_date_and_says_so():
    assert _filing(period_of_report=None).fiscal_period_label() == "filed 2025-10-09"


def test_two_filings_of_one_type_never_share_a_label():
    """financial_metrics is keyed on (ticker, filing_type, fiscal_period)."""
    a = _filing(period_of_report=date(2024, 8, 31)).fiscal_period_label()
    b = _filing(period_of_report=date(2025, 8, 31)).fiscal_period_label()
    assert a != b


# ---------------------------------------------------------------------------
# The command end to end, against stubs
# ---------------------------------------------------------------------------

class _Filings:
    def __init__(self, filings):
        self._filings = filings
        self.marked: list[tuple[int, FilingStatus]] = []

    async def list_by_ticker_and_status(self, ticker, statuses):
        assert FilingStatus.EMBEDDED in statuses
        return self._filings

    async def mark_status(self, filing_id, status, error_message=None):
        self.marked.append((filing_id, status))


class _Metrics:
    def __init__(self):
        self.rows: list[FinancialMetricsRow] = []

    async def upsert(self, row):
        # The real repository type-checks; keep the stub honest about it.
        assert isinstance(row, FinancialMetricsRow), type(row).__name__
        self.rows.append(row)


class _Chunk:
    class chunk:
        id = 1
        ticker = "ACN"
        filing_type = "10-K"
        filed_date = date(2025, 10, 9)
        section_path = ["Part II", "Item 8"]
        content = "Revenue was $64.9 billion."


class _Retrieval:
    def __init__(self, *a, **kw):
        pass

    async def retrieve_for_extraction(self, ticker, filed_date, k=5):
        return [_Chunk()]


class _Extractor:
    def __init__(self):
        self.calls = []

    async def extract(self, chunks, ticker, period, filing_type, filed_date):
        self.calls.append((ticker, period, filing_type, filed_date))
        return _extracted()


@pytest.fixture
def stubbed(monkeypatch):
    filings = _Filings([_filing()])
    metrics = _Metrics()
    extractor = _Extractor()

    async def _noop():
        return None

    monkeypatch.setattr(cli, "init_pool", _noop)
    monkeypatch.setattr(cli, "close_pool", _noop)
    monkeypatch.setattr(cli, "EmbeddingService", lambda *a, **k: object())
    monkeypatch.setattr(cli, "ChunkRepository", lambda *a, **k: object())
    monkeypatch.setattr(cli, "QueryDecomposer", lambda *a, **k: object())
    monkeypatch.setattr(cli, "RetrievalService", _Retrieval)
    monkeypatch.setattr(cli, "MetricsExtractor", lambda *a, **k: extractor)
    monkeypatch.setattr(cli, "MetricsRepository", lambda *a, **k: metrics)
    monkeypatch.setattr(cli, "FilingRepository", lambda *a, **k: filings)
    monkeypatch.setattr(cli, "format_citation_tag", lambda c: "[ACN 10-K 2025 §Item 8]")
    return filings, metrics, extractor


def test_extract_metrics_runs_end_to_end(stubbed):
    filings, metrics, extractor = stubbed
    result = CliRunner().invoke(cli.app, ["extract-metrics", "ACN"])

    assert result.exit_code == 0, result.output + repr(result.exception)
    # The row reached the repository, fully populated.
    assert len(metrics.rows) == 1
    row = metrics.rows[0]
    assert (row.ticker, row.fiscal_period, row.filing_type) == ("ACN", "FY2025", "10-K")
    assert row.source_citations == ["[ACN 10-K 2025 §Item 8]"]
    # The label the extractor was given is the label that was stored.
    assert extractor.calls == [("ACN", "FY2025", "10-K", date(2025, 10, 9))]
    # And the filing was advanced.
    assert filings.marked == [(1, FilingStatus.METRICS_EXTRACTED)]


def test_extract_metrics_says_so_when_the_corpus_is_empty(monkeypatch, stubbed):
    filings, _, _ = stubbed
    monkeypatch.setattr(cli, "FilingRepository", lambda *a, **k: _Filings([]))
    result = CliRunner().invoke(cli.app, ["extract-metrics", "ACN"])
    assert result.exit_code == 1
    assert "No embedded filings" in result.output


def test_a_filing_with_no_chunks_is_skipped_not_stored(monkeypatch, stubbed):
    filings, metrics, _ = stubbed

    class _Empty(_Retrieval):
        async def retrieve_for_extraction(self, ticker, filed_date, k=5):
            return []

    monkeypatch.setattr(cli, "RetrievalService", _Empty)
    result = CliRunner().invoke(cli.app, ["extract-metrics", "ACN"])
    assert result.exit_code == 0, result.output
    assert metrics.rows == []
    assert filings.marked == []
    assert "0 of 1 filing(s) extracted" in result.output
