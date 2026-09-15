"""Failures inside a node that should not take the whole run down with them.

Two gaps from docs/code_review.md (Medium #8):
- synthesizer_node dropped a verdict sample only on the fabrication/reference
  guard errors. Anything else — a second schema failure, a provider error, a
  risk turn raising inside an extra panel — escaped the node and threw away
  every sample already paid for, including ones that had passed.
- The ports' per-node spending caps raised a bare AssertionError, which ended
  the process with no aborted-run artifact and no run summary, while a
  run-level breach got both via graceful_abort.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import app.agent.researcher as researcher
import app.agent.trading.application.debate_nodes as debate_nodes
import app.agent.trading.application.nodes as nodes
import app.agent.trading.application.risk_nodes as risk_nodes
from app.agent.trading.domain.budget import NodeBudgetExceeded, RunTermination
from app.agent.trading.domain.decision_memo import Verdict
from app.agent.trading.infrastructure.graph import build_trading_graph

from tests.agent.trading.test_debate_graph import (
    _run,
    _stub_debate,
    _stub_fundamentals,
    _stub_risk_panel,
    _stub_synthesis,
)
from tests.agent.trading.test_risk_verdict_sampling import (
    _memo,
    _state,
    _stub_extra_panel_samples,
)


def _synthesis_calls(monkeypatch, outcomes):
    """run_synthesis returns or raises outcomes[0], outcomes[1], ... in order."""
    calls = iter(outcomes)

    async def fake(state, *, ledger, base_gaps, base_evidence, as_of, client=None):
        outcome = next(calls)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(nodes, "run_synthesis", fake)


# ---------------------------------------------------------------------------
# A failed sample is one lost vote, not a lost run
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_a_sample_that_errors_is_dropped_and_the_rest_still_vote(monkeypatch):
    _stub_extra_panel_samples(monkeypatch)
    _synthesis_calls(monkeypatch, [
        _memo("ACN", Verdict.HOLD, tag="a"),
        RuntimeError("provider returned 500"),
        _memo("ACN", Verdict.HOLD, tag="c"),
    ])

    memo = (await nodes.synthesizer_node(_state()))["decision_memo"]

    assert memo.verdict == Verdict.HOLD
    assert memo.verdict_samples == ["hold", "hold"]
    assert any("1 of 3 risk-verdict sample(s) failed with an error" in g for g in memo.data_gaps)
    assert any("provider returned 500" in g for g in memo.data_gaps)


@pytest.mark.anyio
async def test_an_extra_panel_that_raises_drops_only_its_own_sample(monkeypatch):
    """The extra panel's risk turns ran OUTSIDE the old try block, so one bad
    turn in sample 2 killed samples 1 and 3 as well."""
    calls = {"n": 0}

    async def flaky_turn(state, persona, turn_index):
        calls["n"] += 1
        if calls["n"] == 1:   # first turn of the first extra panel
            raise ValueError("risk turn schema violation after retry")
        from tests.agent.trading.test_risk_verdict_sampling import _turn
        return _turn(turn_index, persona, propose=(turn_index == 0))

    monkeypatch.setattr(risk_nodes, "run_risk_turn", flaky_turn)
    _synthesis_calls(monkeypatch, [
        _memo("ACN", Verdict.SELL, tag="a"),
        _memo("ACN", Verdict.SELL, tag="c"),
    ])

    memo = (await nodes.synthesizer_node(_state()))["decision_memo"]

    assert memo.verdict_samples == ["sell", "sell"]
    assert any("ValueError: risk turn schema violation" in g for g in memo.data_gaps)


@pytest.mark.anyio
async def test_when_every_sample_fails_the_node_raises_one_clear_error(monkeypatch):
    _stub_extra_panel_samples(monkeypatch)
    _synthesis_calls(monkeypatch, [RuntimeError("a"), RuntimeError("b"), RuntimeError("c")])

    with pytest.raises(RuntimeError, match="no risk-verdict sample for ACN survived"):
        await nodes.synthesizer_node(_state())


@pytest.mark.anyio
async def test_a_node_budget_breach_in_a_sample_is_not_swallowed_as_a_failed_vote(monkeypatch):
    _stub_extra_panel_samples(monkeypatch)
    _synthesis_calls(monkeypatch, [
        _memo("ACN", Verdict.HOLD, tag="a"),
        NodeBudgetExceeded("synthesis cost $0.40 exceeds the $0.30 budget"),
    ])

    with pytest.raises(NodeBudgetExceeded):
        await nodes.synthesizer_node(_state())


# ---------------------------------------------------------------------------
# A node's own cap ends the run gracefully
# ---------------------------------------------------------------------------

def test_the_breach_is_still_an_assertion_error_for_existing_callers():
    assert issubclass(NodeBudgetExceeded, AssertionError)


@pytest.mark.anyio
async def test_a_debate_turn_over_its_cap_aborts_the_run_with_an_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)
    _stub_fundamentals(monkeypatch)
    _stub_risk_panel(monkeypatch)
    _stub_synthesis(monkeypatch)
    _stub_debate(monkeypatch)
    real_turn = debate_nodes.run_debate_turn

    async def capped_turn(state, side, turn_index):
        if turn_index == 2:
            raise NodeBudgetExceeded("debate cost $0.40 for ACN exceeds the $0.35 per-debate budget")
        return await real_turn(state, side, turn_index)

    monkeypatch.setattr(debate_nodes, "run_debate_turn", capped_turn)

    result, _ = await _run(build_trading_graph(InMemorySaver(), analysts=["fundamentals"]))

    assert result["run_terminated_by"] == RunTermination.NODE_BUDGET_EXCEEDED
    assert "per-debate budget" in result["node_budget_breach"]
    assert len(result["debate_turns"]) == 2          # turns 0 and 1; nothing after
    assert result.get("risk_turns") in (None, [])    # the panel never ran
    assert result.get("decision_memo") is None
    [aborted] = list(tmp_path.rglob("*decision-ABORTED*.md"))
    text = aborted.read_text()
    assert "NODE_BUDGET_EXCEEDED" in text
    assert "per-debate budget" in text


@pytest.mark.anyio
async def test_the_synthesizer_over_its_cap_aborts_instead_of_ending_without_a_memo(monkeypatch, tmp_path):
    """The synthesizer is the last node; its exit edge used to be a plain
    edge to END, so a breach there would have ended the run with no memo
    and no abort record."""
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)
    _stub_fundamentals(monkeypatch)
    _stub_risk_panel(monkeypatch)
    _stub_debate(monkeypatch)

    async def capped(state, *, ledger, base_gaps, base_evidence, as_of, client=None):
        raise NodeBudgetExceeded("synthesis cost $0.40 for ACN exceeds the $0.30 budget")

    monkeypatch.setattr(nodes, "run_synthesis", capped)

    result, _ = await _run(build_trading_graph(InMemorySaver(), analysts=["fundamentals"]))

    assert result["run_terminated_by"] == RunTermination.NODE_BUDGET_EXCEEDED
    assert result.get("decision_memo") is None
    assert list(tmp_path.rglob("*decision-ABORTED*.md"))


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# A price vendor being down is not a reason to lose the run
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_a_vendor_outage_degrades_the_technical_leg_instead_of_ending_the_run(monkeypatch):
    """Live, FIG 2026-09-13: yfinance returned an empty frame and Finnhub's
    free tier 403'd on historical candles. VendorError propagated out of the
    node and ended the run — discarding the fundamentals leg that had
    already completed and been paid for ($0.070). Nothing about a price feed
    being down invalidates the filing analysis."""
    from datetime import date

    import app.agent.trading.application.nodes as nodes
    from app.agent.trading.domain.errors import VendorError

    async def boom(ticker, as_of):
        raise VendorError("No price data for FIG from yfinance or Finnhub")

    monkeypatch.setattr(nodes, "get_price_history", boom)

    out = await nodes.technical_node({"ticker": "FIG", "as_of_date": date(2026, 3, 1)})

    assert out == {"analyst_failures": ["technical: No price data for FIG from yfinance or Finnhub"]}
    assert "technical_report" not in out       # nothing fabricated to fill the hole


@pytest.mark.parametrize("missing,failures,expect", [
    (["technical"], ["technical: No price data for FIG from yfinance or Finnhub"],
     ["FAILED", "yfinance"]),
    (["technical"], [], ["did not run"]),
])
def test_a_failed_analyst_and_an_unselected_one_read_differently(missing, failures, expect):
    """Same missing section, different claim about it. An analyst nobody
    selected cannot claim a vendor was down; one whose vendor WAS down must
    not read as though the evidence were merely unrequested."""
    import app.agent.trading.application.nodes as nodes

    gap = nodes._missing_analyst_gaps(missing, failures)[0]
    for token in expect:
        assert token in gap
    assert "not the same as that evidence being neutral" in gap or "FAILED" in gap


def test_a_malformed_failure_entry_falls_back_to_did_not_run():
    """Never crash the memo over the shape of a diagnostic string."""
    import app.agent.trading.application.nodes as nodes

    assert "did not run" in nodes._missing_analyst_gaps(["technical"], ["technical"])[0]
    assert "did not run" in nodes._missing_analyst_gaps(["technical"], ["technical: "])[0]
