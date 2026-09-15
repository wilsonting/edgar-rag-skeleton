"""The local API's exposure: who can call it, and what a ticker can do.

Found in review (docs/code_review.md, Medium #4):
- CORS allowed every origin, so any page open in the user's browser could
  call /ask, /extract, /ingest, /news-assess and /trading/analyze, each of
  which spends API credits.
- Nothing authenticated those endpoints.
- The ticker went unvalidated into `MEMO_DIR / ticker`, so "../../TMP/X"
  wrote outside the vault (case-insensitive filesystems ignore the upper()).
"""

from __future__ import annotations

import argparse

import pytest
from fastapi.testclient import TestClient

import app.agent.researcher as researcher
import app.agent.tools as tools
import app.main as main
from app.domain.values import normalize_ticker


# ---------------------------------------------------------------------------
# The ticker rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("acn", "ACN"), (" ACN ", "ACN"), ("BRK.B", "BRK.B"), ("BF-B", "BF-B"), ("0700", "0700"),
])
def test_real_tickers_pass_and_are_normalized(raw, expected):
    assert normalize_ticker(raw) == expected


@pytest.mark.parametrize("raw", ["..", ".", "../../TMP/X", "A/B", "", "-X", "TOOLONGTICKER1", "A B"])
def test_anything_that_could_be_a_path_or_is_not_a_ticker_is_rejected(raw):
    with pytest.raises(ValueError):
        normalize_ticker(raw)


def test_the_vault_writer_refuses_a_bad_ticker_even_if_an_entry_point_let_it_through(tmp_path, monkeypatch):
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path / "vault")
    with pytest.raises(ValueError):
        researcher._save_output("memo", "../../escaped", "fundamentals")
    assert not any(tmp_path.rglob("*.md"))


def test_the_cli_rejects_a_bad_ticker_as_a_usage_error():
    with pytest.raises(argparse.ArgumentTypeError):
        researcher._ticker_arg("../x")
    assert researcher._ticker_arg("acn") == "ACN"


# ---------------------------------------------------------------------------
# Entry points reject a bad ticker before doing anything
# ---------------------------------------------------------------------------

class _ExplodingGraph:
    async def aget_state(self, config):
        raise AssertionError("the graph must not be touched for an invalid ticker")

    ainvoke = aget_state


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("APP_API_KEY", raising=False)
    monkeypatch.setattr(main.app.state, "trading_graph", _ExplodingGraph(), raising=False)
    return TestClient(main.app)


def test_trading_analyze_rejects_a_path_as_a_ticker(client):
    resp = client.post("/trading/analyze", json={"ticker": "../../TMP/X"})
    assert resp.status_code == 422


def test_ask_rejects_a_bad_ticker_filter(client):
    resp = client.post("/ask", json={"question": "revenue?", "tickers": ["../x"]})
    assert resp.status_code == 422


def test_corpus_status_rejects_a_bad_ticker(client):
    resp = client.get("/corpus-status", params={"ticker": "../x"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

def _preflight(client, origin):
    return client.options("/ask", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })


def test_an_arbitrary_website_is_not_allowed_to_call_the_api(client):
    resp = _preflight(client, "https://attacker.example")
    assert resp.headers.get("access-control-allow-origin") is None


def test_the_local_dev_ui_is_allowed(client):
    resp = _preflight(client, "http://localhost:5173")
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_the_default_allow_list_is_only_the_vite_dev_origins():
    # Read at import; the suite runs without CORS_ALLOW_ORIGINS set.
    assert main.CORS_ALLOW_ORIGINS == ["http://localhost:5173", "http://127.0.0.1:5173"]


# ---------------------------------------------------------------------------
# Opt-in shared secret on the endpoints that spend money
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("headers,status", [({}, 401), ({"X-API-Key": "wrong"}, 401)])
def test_with_a_key_configured_spending_endpoints_need_it(client, monkeypatch, headers, status):
    monkeypatch.setenv("APP_API_KEY", "s3cret")
    resp = client.post("/trading/analyze", json={"ticker": "ACN"}, headers=headers)
    assert resp.status_code == status


def test_the_right_key_gets_through(client, monkeypatch):
    """Past auth the request reaches the graph — here the exploding one, so
    reaching it at all is the proof."""
    monkeypatch.setenv("APP_API_KEY", "s3cret")
    with pytest.raises(AssertionError, match="must not be touched"):
        client.post("/trading/analyze", json={"ticker": "ACN"}, headers={"X-API-Key": "s3cret"})


def test_read_only_endpoints_do_not_need_the_key(client, monkeypatch):
    monkeypatch.setenv("APP_API_KEY", "s3cret")
    # Rejected on the ticker (422), not on auth (401): the auth check does
    # not apply to /corpus-status.
    assert client.get("/corpus-status", params={"ticker": "../x"}).status_code == 422


@pytest.mark.anyio
async def test_the_agents_tool_calls_send_the_key_when_one_is_configured(monkeypatch):
    seen = {}

    class _Resp:
        status_code = 200
        headers: dict = {}
        text = "{}"

    class _Client:
        def __init__(self, **kw):
            seen.update(kw.get("headers") or {})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None, **kw):
            return _Resp()

    monkeypatch.setattr(tools.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(tools, "USE_STUBS", False)
    monkeypatch.setenv("APP_API_KEY", "s3cret")

    await tools._dispatch("check_corpus", {"ticker": "ACN"})

    assert seen == {"X-API-Key": "s3cret"}


@pytest.fixture
def anyio_backend():
    return "asyncio"
