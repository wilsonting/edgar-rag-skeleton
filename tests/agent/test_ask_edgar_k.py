"""`ask_edgar` retrieves ASK_EDGAR_K chunks per call, and that number is
~77% of the call's input cost. It is also the knob most likely to be turned
for a saving without measuring what it costs, so the measurement lives next
to it — see scripts/probe_retrieval_rank_decay.py and the constant's comment.
"""

from __future__ import annotations

import pytest

import app.agent.tools as tools


@pytest.fixture(autouse=True)
def _fresh():
    tools.reset_run_provenance()
    yield
    tools.reset_run_provenance()


def test_k_defaults_to_the_measured_value_not_the_cheaper_one():
    """5 was tried and reverted: it saves ~$0.05/run and drops a filing
    section nothing else retrieved on 59% of questions. The default has to
    be the measured-correct value, not the cheap one — a default of 5 with
    an env override to 8 would give every fresh checkout and CI run the
    version the measurement rejected."""
    import os
    assert "ASK_EDGAR_K" in os.environ or tools.ASK_EDGAR_K == 8


@pytest.mark.anyio
async def test_the_request_sends_k_explicitly(monkeypatch):
    """The endpoint's own default is 8. Relying on it would mean the agent's
    retrieval width silently tracked an unrelated API default."""
    sent = {}

    class _Resp:
        status_code = 200
        headers: dict = {}
        text = "{}"

        def json(self):
            return {"answer": "a", "chunks": []}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, **kw):
            sent.update(json or {})
            return _Resp()

    monkeypatch.setattr(tools.httpx, "AsyncClient", lambda **kw: _Client())
    monkeypatch.setattr(tools, "USE_STUBS", False)

    await tools._dispatch("ask_edgar", {"question": "q", "tickers": ["NFLX"]})

    assert sent["k"] == tools.ASK_EDGAR_K
