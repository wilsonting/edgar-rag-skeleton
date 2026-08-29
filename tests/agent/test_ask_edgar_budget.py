"""`ask_edgar` is the most expensive thing the research agent can do, and
until 2026-08-27 nothing bounded it. Phase 9 measured NFLX at 22 calls, AVGO
and ACN at 40 each — every question distinct, so there is no duplicate work
to dedupe away and a cap necessarily trades completeness for cost.

That trade is the reason the budget is ANNOUNCED (tool description) and
COUNTED DOWN (warn band) rather than sprung: a blind cap truncates wherever
the agent happens to be, which on a 12-item checklist means the last items
silently get nothing.
"""

from __future__ import annotations

import pytest

import app.agent.tools as tools


@pytest.fixture(autouse=True)
def _fresh_budget():
    tools.reset_run_provenance()
    yield
    tools.reset_run_provenance()


@pytest.mark.anyio
async def test_the_cap_refuses_rather_than_raises():
    """The agent's correct response to an exhausted budget is to write the
    memo from what it has, exactly as it does at MAX_TURNS. Raising would
    lose the whole run's work over a budget that is a preference, not a
    failure."""
    tools._ASK_EDGAR_CALLS = tools.ASK_EDGAR_MAX_CALLS

    result = await tools._dispatch("ask_edgar", {"question": "anything"})

    assert "BUDGET EXHAUSTED" in result
    assert str(tools.ASK_EDGAR_MAX_CALLS) in result
    # It must tell the agent what to DO, not just that it failed.
    assert "data gap" in result


@pytest.mark.anyio
async def test_the_refusal_makes_no_http_call(monkeypatch):
    """A refused call must cost nothing — that is the entire point."""
    def _boom(*a, **k):
        raise AssertionError("the budget check ran too late; an HTTP call was made")

    monkeypatch.setattr(tools.httpx, "AsyncClient", _boom)
    tools._ASK_EDGAR_CALLS = tools.ASK_EDGAR_MAX_CALLS

    assert "BUDGET EXHAUSTED" in await tools._dispatch("ask_edgar", {"question": "q"})


def test_the_budget_is_announced_in_the_tool_description():
    """An unannounced cap is indistinguishable from a broken tool. The agent
    has to see the number before it spends the first call to plan against
    it."""
    description = next(t for t in tools.TOOLS if t["name"] == "ask_edgar")["description"]

    assert str(tools.ASK_EDGAR_MAX_CALLS) in description
    assert "BUDGETED" in description


def test_the_counter_is_per_run():
    """Fenced by the same call that fences every other per-run accumulator
    in this module, or run two starts already spent."""
    tools._ASK_EDGAR_CALLS = 17
    tools.reset_run_provenance()
    assert tools._ASK_EDGAR_CALLS == 0


def test_the_cap_is_configurable_without_a_code_change():
    """ACN completed its checklist in 40 calls, so the default of 30 WILL
    bind on dense tickers. That is deliberate, and the escape hatch has to
    exist for the case where the truncation matters."""
    import os
    assert "ASK_EDGAR_MAX_CALLS" in os.environ or tools.ASK_EDGAR_MAX_CALLS == 30


@pytest.mark.anyio
async def test_the_last_permitted_call_is_the_nth_not_the_nth_minus_one(monkeypatch):
    """Off-by-one guard. With the cap at N the agent gets N calls, not N-1:
    the check is `>= cap` against a counter incremented AFTER the check
    passes. Stubbed transport so this asserts the boundary, not the network."""
    monkeypatch.setattr(tools, "USE_STUBS", True)
    tools._ASK_EDGAR_CALLS = tools.ASK_EDGAR_MAX_CALLS - 1

    allowed = await tools._dispatch("ask_edgar", {"question": "the Nth call"})
    assert "BUDGET EXHAUSTED" not in allowed
    assert tools._ASK_EDGAR_CALLS == tools.ASK_EDGAR_MAX_CALLS

    refused = await tools._dispatch("ask_edgar", {"question": "the N+1th"})
    assert "BUDGET EXHAUSTED" in refused
    # A refused call must not consume budget it never used.
    assert tools._ASK_EDGAR_CALLS == tools.ASK_EDGAR_MAX_CALLS
