"""The run budget and deadline, applied INSIDE the nodes that make many calls.

`check_run_guards` ran only on graph edges, so the fundamentals node — up to
LOOP_MAX_TURNS model calls plus every server-side call its tools make — and
the synthesizer's extra verdict samples (each a fresh 9-turn risk panel plus
two synthesis calls) spent unseen until the node returned. The CLI's own
comment conceded it: "the run-level guards can only fire between nodes,
which on this graph means after the fundamentals stage has already been
paid for." These pin the in-node checks.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import app.agent.researcher as researcher
import app.agent.trading.application.nodes as nodes
import app.agent.trading.infrastructure.fundamentals_port as port
from app.agent.trading.domain.budget import CostEvent, RunBudget
from app.agent.trading.domain.decision_memo import Verdict
from app.domain.token_usage import TokenUsage

from tests.agent.trading.test_risk_verdict_sampling import (
    _memo,
    _state,
    _stub_extra_panel_samples,
    _stub_synthesis_sequence,
)


def _event(usd: float, event_id: str = "earlier:1") -> CostEvent:
    return CostEvent(
        event_id=event_id, node="earlier", model="m", input_tokens=0, output_tokens=0,
        cache_creation_input_tokens=0, cache_read_input_tokens=0, usd=usd,
    )


def _budget(max_usd: float = 1.0, *, minutes: float = 30) -> RunBudget:
    return RunBudget(
        max_usd=max_usd, deadline_utc=datetime.now(timezone.utc) + timedelta(minutes=minutes)
    )


# ---------------------------------------------------------------------------
# run_agent's stop check
# ---------------------------------------------------------------------------

def _usage(out: int = 10):
    return SimpleNamespace(
        input_tokens=0, output_tokens=out,
        cache_creation_input_tokens=0, cache_read_input_tokens=0,
    )


def _tool_use(i: int):
    return SimpleNamespace(type="tool_use", id=f"t{i}", name="check_corpus", input={"ticker": "ACN"})


def _text(content) -> str:
    """A turn's text, whether it is a bare string or content blocks.

    Every call now goes through `_roll_cache_breakpoint`, which wraps a
    string turn so the breakpoint has a block to attach to.
    """
    if isinstance(content, str):
        return content
    return " ".join(
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
    )


class FakeAgentClient:
    """Answers every turn with tool calls, and the forced-memo turn (the one
    whose last message is the budget note) with a memo."""

    def __init__(self, tools_per_turn: int = 1):
        self.tools_per_turn = tools_per_turn
        self.calls: list[list[dict]] = []
        self.requests: list[dict] = []
        self.messages = self

    async def create(self, **kwargs):
        messages = kwargs["messages"]
        self.calls.append(messages)
        self.requests.append(kwargs)
        last = _text(messages[-1]["content"])
        if "Write the memo now" in last:
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="# ACN memo\n## Assessment\nstub")],
                stop_reason="end_turn", usage=_usage(),
            )
        n = len(self.calls)
        return SimpleNamespace(
            content=[_tool_use(n * 10 + k) for k in range(self.tools_per_turn)],
            stop_reason="tool_use", usage=_usage(),
        )


@pytest.fixture
def agent(monkeypatch):
    client = FakeAgentClient()
    executed: list[str] = []

    async def fake_execute(name, inputs):
        executed.append(name)
        return "ok"

    monkeypatch.setattr(researcher, "get_client", lambda model: client)
    monkeypatch.setattr(researcher, "execute_tool", fake_execute)
    return SimpleNamespace(client=client, executed=executed)


@pytest.mark.anyio
async def test_the_loop_stops_when_the_check_fires_and_still_writes_a_memo(agent):
    # Each model call reports 10 output tokens. The check allows turn 1 and
    # its tool (usage 10), then fires at usage 20: turn 2's tool call is
    # refused, and the next turn never starts — the forced memo runs instead.
    stop = lambda usage: "budget_exceeded" if usage.output_tokens >= 20 else None

    memo, usage = await researcher.run_agent("task", "system", stop_check=stop)

    assert memo.startswith("# ACN memo")
    assert len(agent.client.calls) == 3        # turn 1, turn 2, forced memo
    assert agent.executed == ["check_corpus"]  # turn 2's tool was refused
    refused = agent.client.calls[2][-2]["content"][0]["content"]
    assert refused.startswith("RUN BUDGET REACHED: budget_exceeded")
    assert "spending or time budget has been reached (budget_exceeded)" in _text(
        agent.client.calls[2][-1]["content"]
    )
    assert usage.output_tokens == 30


@pytest.mark.anyio
async def test_a_breach_before_the_first_turn_spends_only_the_memo_call(agent):
    await researcher.run_agent("task", "system", stop_check=lambda usage: "deadline_exceeded")

    assert len(agent.client.calls) == 1
    assert agent.executed == []


@pytest.mark.anyio
async def test_a_breach_mid_turn_refuses_the_remaining_tool_calls(agent, monkeypatch):
    """One turn can carry several tool calls (gpt-5.6-luna sent 6-7
    ask_edgar calls per turn on 2026-09-11), each with its own server-side
    spend — so the check runs before each wave of calls, not only per turn.
    At concurrency 1 a wave is one call: the strictest setting."""
    monkeypatch.setattr(researcher, "TOOL_CONCURRENCY", 1)
    agent.client.tools_per_turn = 3
    stop = lambda usage: "budget_exceeded" if agent.executed else None

    await researcher.run_agent("task", "system", stop_check=stop)

    assert agent.executed == ["check_corpus"]
    results = agent.client.calls[-1][-2]["content"]
    assert [r["content"] == "ok" for r in results] == [True, False, False]
    assert all("RUN BUDGET REACHED" in r["content"] for r in results[1:])


@pytest.mark.anyio
async def test_without_a_check_the_loop_is_unchanged(agent, monkeypatch):
    monkeypatch.setattr(researcher, "MAX_TURNS", 2)

    await researcher.run_agent("task", "system")

    assert len(agent.client.calls) == 3        # both turns, then the MAX_TURNS memo
    assert agent.executed == ["check_corpus", "check_corpus"]
    assert "exhausted your tool-call budget" in _text(agent.client.calls[-1][-1]["content"])


# ---------------------------------------------------------------------------
# fundamentals_port.budget_stop_check
# ---------------------------------------------------------------------------

def _agent_usage(input_tokens: int = 0):
    u = researcher.UsageSummary()
    u.input_tokens = input_tokens
    return u


def test_no_budget_means_no_check():
    assert port.budget_stop_check(None) is None


def test_under_budget_the_check_passes(monkeypatch):
    monkeypatch.setattr(port, "get_delegated_usage", lambda: {})
    check = port.budget_stop_check(_budget(1.0), [_event(0.10)])
    assert check(_agent_usage(1_000)) is None


def test_delegated_spend_is_priced_at_its_own_model_and_counts(monkeypatch):
    """1M DeepSeek input tokens are $0.44 at the table's rate — priced as
    DeepSeek regardless of the agent's model, like the final accounting."""
    monkeypatch.setattr(port, "get_delegated_usage", lambda: {
        "deepseek-v4-flash": TokenUsage(input_tokens=1_000_000),
    })
    under = port.budget_stop_check(_budget(1.0), [_event(0.50)])
    over = port.budget_stop_check(_budget(1.0), [_event(0.60)])

    assert under(_agent_usage()) is None                 # 0.50 + 0.44 < 1.00
    reason = over(_agent_usage())                        # 0.60 + 0.44 >= 1.00
    assert reason.startswith("budget_exceeded")
    assert "$1.00 run budget" in reason


def test_a_passed_deadline_stops_the_loop(monkeypatch):
    monkeypatch.setattr(port, "get_delegated_usage", lambda: {})
    check = port.budget_stop_check(_budget(10.0, minutes=-1))
    assert check(_agent_usage()).startswith("deadline_exceeded")


@pytest.mark.anyio
async def test_the_fundamentals_node_hands_the_port_its_budget_and_prior_spend(monkeypatch):
    seen = {}

    async def fake_report(ticker, as_of=None, run_id=None, **kwargs):
        seen["as_of"] = as_of
        seen.update(kwargs)
        return None

    monkeypatch.setattr(nodes, "get_fundamentals_report", fake_report)
    budget, prior = _budget(0.75), [_event(0.05)]

    as_of = date(2026, 8, 19)
    await nodes.fundamentals_node(
        {"ticker": "ACN", "as_of_date": as_of, "budget": budget, "cost_events": prior}
    )

    # The analysis date rides with the budget: the node cannot pass one and
    # forget the other.
    assert seen == {"as_of": as_of, "budget": budget, "prior_events": prior}


# ---------------------------------------------------------------------------
# synthesizer_node's extra samples
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_extra_verdict_samples_are_skipped_once_the_budget_is_spent(monkeypatch):
    _stub_extra_panel_samples(monkeypatch)
    first = _memo("ACN", Verdict.HOLD, tag="a").model_copy(
        update={"cost_events": [_event(0.06, "synthesis:1")]}
    )
    _stub_synthesis_sequence(monkeypatch, [first, _memo("ACN", Verdict.SELL, tag="b")])

    # 0.05 before the node + 0.06 for sample 1 = 0.11 >= 0.10
    state = _state(budget=_budget(0.10), cost_events=[_event(0.05)])
    memo = (await nodes.synthesizer_node(state))["decision_memo"]

    assert memo.verdict_samples == ["hold"]
    assert any("2 of 3 risk-verdict sample(s) were skipped" in g for g in memo.data_gaps)
    assert any("budget exceeded" in g for g in memo.data_gaps)


@pytest.mark.anyio
async def test_under_budget_all_samples_still_run(monkeypatch):
    _stub_extra_panel_samples(monkeypatch)
    _stub_synthesis_sequence(monkeypatch, [
        _memo("ACN", Verdict.HOLD, tag="a"),
        _memo("ACN", Verdict.SELL, tag="b"),
        _memo("ACN", Verdict.HOLD, tag="c"),
    ])

    memo = (await nodes.synthesizer_node(_state(budget=_budget(5.0))))["decision_memo"]

    assert memo.verdict_samples == ["hold", "sell", "hold"]
    assert not any("skipped" in g for g in memo.data_gaps)


@pytest.fixture
def anyio_backend():
    return "asyncio"
