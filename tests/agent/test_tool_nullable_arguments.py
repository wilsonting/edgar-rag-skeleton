"""Tool arguments and what the server actually does with them.

Strict tool calling makes every optional argument required-but-nullable, so
a model that leaves one unset sends an explicit null. `ingest_ticker` read
its limit as `inputs.get("limit", 3)`, which returns None for that, and
/ingest's `limit: int` rejected the request with a 422. Separately,
`extract_metrics` offered `filed_after`/`filed_before` to the agent and
/extract ignored both.
"""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

import app.agent.tools as tools
import app.main as main
from app.application.extraction_service import FinancialMetrics


class _Resp:
    status_code = 200
    headers: dict = {}
    text = '{"status": "ok"}'

    def json(self):
        return {"status": "ok"}


@pytest.fixture
def posted(monkeypatch):
    sent: list[tuple[str, dict]] = []

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, **kw):
            sent.append((url, json))
            return _Resp()

    monkeypatch.setattr(tools.httpx, "AsyncClient", lambda **kw: _Client())
    monkeypatch.setattr(tools, "USE_STUBS", False)
    return sent


@pytest.mark.anyio
async def test_ingest_ticker_with_a_null_limit_sends_the_default(posted):
    await tools._dispatch("ingest_ticker", {"ticker": "ACN", "limit": None, "form_type": None})

    [(url, payload)] = posted
    assert url.endswith("/ingest")
    assert payload == {"ticker": "ACN", "limit": 3}


@pytest.mark.anyio
async def test_ingest_ticker_passes_explicit_arguments_through(posted):
    await tools._dispatch("ingest_ticker", {"ticker": "ACN", "limit": 5, "form_type": "10-Q"})

    [(_, payload)] = posted
    assert payload == {"ticker": "ACN", "limit": 5, "form_type": "10-Q"}


# ---------------------------------------------------------------------------
# /extract honours the window the tool offers
# ---------------------------------------------------------------------------

@pytest.fixture
def extract_client(monkeypatch):
    """Everything behind /extract that would touch OpenAI or Postgres is
    replaced; what is left is the endpoint's own window logic."""
    windows: list[tuple[date, date]] = []

    async def fake_gather(retrieval, ticker, filed_after, filed_before):
        windows.append((filed_after, filed_before))
        return []

    class _Extractor:
        llm_model = "deepseek-v4-flash"
        last_usage = main.TokenUsage()

        async def extract(self, chunks, ticker, period, filing_type, filed_date):
            return FinancialMetrics(
                revenue=None, gross_margin_pct=None, gaap_net_income=None,
                free_cash_flow=None, sbc_pct_of_revenue=None, net_dollar_retention=None,
                extraction_confidence="not_disclosed",
            )

    class _Repo:
        def __init__(self, *a, **k):
            pass

        async def upsert(self, row):
            pass

    for name, value in {
        "gather_extraction_chunks": fake_gather,
        "EmbeddingService": lambda: None,
        "ChunkRepository": lambda: None,
        "QueryDecomposer": lambda: None,
        "MetricsExtractor": _Extractor,
        "MetricsRepository": _Repo,
    }.items():
        monkeypatch.setattr(main, name, value)
    return TestClient(main.app), windows


def _extract(client, **extra):
    return client.post("/extract", json={
        "ticker": "ACN", "fiscal_period": "FY2025", "filing_type": "10-K",
        "filed_date": "2025-10-10", **extra,
    })


def test_extract_defaults_to_thirty_days_either_side(extract_client):
    client, windows = extract_client
    assert _extract(client).status_code == 200
    assert windows == [(date(2025, 9, 10), date(2025, 11, 9))]


def test_extract_uses_the_bounds_it_is_given(extract_client):
    client, windows = extract_client
    resp = _extract(client, filed_after="2025-10-10", filed_before="2025-10-10")
    assert resp.status_code == 200
    assert windows == [(date(2025, 10, 10), date(2025, 10, 10))]


def test_extract_treats_null_bounds_as_unset(extract_client):
    client, windows = extract_client
    assert _extract(client, filed_after=None, filed_before=None).status_code == 200
    assert windows == [(date(2025, 9, 10), date(2025, 11, 9))]


def test_extract_rejects_an_inverted_window(extract_client):
    client, windows = extract_client
    resp = _extract(client, filed_after="2025-12-01", filed_before="2025-10-01")
    assert resp.status_code == 400
    assert windows == []


@pytest.fixture
def anyio_backend():
    return "asyncio"
