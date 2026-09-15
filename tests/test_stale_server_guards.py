"""Two guards against the failure that wasted a verification run.

On 2026-09-13 a 22-hour-old `uvicorn` served a full FIG pipeline run. It
predated every fix the run existed to verify. The only symptom was a
`filed_before` bound the old server silently discarded — Pydantic drops
unknown fields by default, so the request succeeded and the bound did
nothing. The run completed, looked clean, and verified nothing.

Two independent things had to be true for that to be silent, so both are
closed here: the server never said what code it was running, and it accepted
a field it did not understand.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app.agent.tools as tools
import app.main as main


# ---------------------------------------------------------------------------
# A field the server does not understand is a 422, not a shrug
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,good", [
    ("AskRequest", {"question": "q"}),
    ("ExtractRequest", {"ticker": "FIG", "fiscal_period": "FY2025",
                        "filing_type": "10-K", "filed_date": "2026-02-18"}),
    ("NewsAssessRequest", {"ticker": "FIG", "headline": "h"}),
    ("IngestRequest", {"ticker": "FIG"}),
    ("LatestFilingsRequest", {"ticker": "FIG"}),
    ("TradingAnalysisRequest", {"ticker": "FIG"}),
])
def test_every_request_model_rejects_an_unknown_field(model, good):
    cls = getattr(main, model)
    cls(**good)                                  # the happy path still works
    with pytest.raises(ValidationError):
        cls(**good, no_such_field="x")


def test_the_exact_field_that_was_silently_dropped():
    """A stale server had no `filed_before` on this model, took the request
    anyway, and threw the bound away."""
    ok = main.LatestFilingsRequest(ticker="FIG", filed_before="2026-03-01")
    assert ok.filed_before.isoformat() == "2026-03-01"
    with pytest.raises(ValidationError) as exc:
        main.LatestFilingsRequest(ticker="FIG", filed_befor="2026-03-01")
    assert "filed_befor" in str(exc.value)


# ---------------------------------------------------------------------------
# The server says what it is running
# ---------------------------------------------------------------------------

def test_health_reports_the_running_commit():
    with TestClient(main.app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "commit" in body and "dirty" in body


def test_the_commit_is_snapshotted_at_import_not_recomputed():
    """The value must be what the PROCESS started with. Recomputed per
    request it would track the working tree and always agree, which is
    exactly the comparison that needs to be able to fail."""
    import inspect

    from app.infrastructure import build_info as bi

    source = inspect.getsource(bi)
    assert "COMMIT: str | None = _head()" in source
    assert bi.build_info()["commit"] == bi.COMMIT


def test_git_being_unavailable_is_not_an_error(monkeypatch):
    from app.infrastructure import build_info as bi

    monkeypatch.setattr(bi, "_git", lambda *a: None)
    assert bi._head() is None
    assert bi.working_tree_matches()[0] is True      # nothing to compare


# ---------------------------------------------------------------------------
# The client notices
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p


class _Http:
    def __init__(self, resp):
        self._resp = resp

    async def get(self, url, **kw):
        if self._resp is None:
            raise RuntimeError("no /health on this server")
        return self._resp


@pytest.fixture(autouse=True)
def _reset():
    tools._SERVER_VERSION_CHECKED = False
    yield
    tools._SERVER_VERSION_CHECKED = False


@pytest.mark.anyio
async def test_a_mismatched_server_is_called_out(monkeypatch, caplog):
    monkeypatch.setattr("app.infrastructure.build_info.COMMIT", "aaaaaaa")
    with caplog.at_level(logging.WARNING, logger=tools.logger.name):
        await tools._warn_if_server_is_stale(_Http(_Resp({"commit": "bbbbbbb"})))
    assert "running bbbbbbb" in caplog.text and "aaaaaaa" in caplog.text
    assert "Restart it" in caplog.text


@pytest.mark.anyio
async def test_a_server_with_no_health_endpoint_is_called_out(monkeypatch, caplog):
    """An older server has no /health, and that IS the answer."""
    with caplog.at_level(logging.WARNING, logger=tools.logger.name):
        await tools._warn_if_server_is_stale(_Http(None))
    assert "predates /health" in caplog.text


@pytest.mark.anyio
async def test_a_matching_server_is_silent(monkeypatch, caplog):
    monkeypatch.setattr("app.infrastructure.build_info.COMMIT", "aaaaaaa")
    with caplog.at_level(logging.WARNING, logger=tools.logger.name):
        await tools._warn_if_server_is_stale(_Http(_Resp({"commit": "aaaaaaa"})))
    assert caplog.text == ""


@pytest.mark.anyio
async def test_the_check_runs_once_per_process_not_per_tool_call(monkeypatch):
    """30 ask_edgar calls must not mean 30 probes."""
    calls = []

    class _Counting(_Http):
        async def get(self, url, **kw):
            calls.append(url)
            return _Resp({"commit": "aaaaaaa"})

    monkeypatch.setattr("app.infrastructure.build_info.COMMIT", "aaaaaaa")
    http = _Counting(None)
    for _ in range(5):
        await tools._warn_if_server_is_stale(http)
    assert len(calls) == 1
