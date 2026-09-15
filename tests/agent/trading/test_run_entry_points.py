"""The run lifecycle, and the two entry points that share it.

`POST /trading/analyze` used to invoke the graph with `{"ticker": ...}` and
nothing else. The fundamentals node ran — the most expensive one — and the
technical node then raised on the missing `as_of_date`: a 500 after the
spend, with no budget guard, since `_guarded()` reads a missing budget as
"opted out". A run that aborted on its budget also crashed the response on
`result["decision_memo"]`. Nothing tested any endpoint, so none of it
showed. These tests pin the state a new run starts from, and that the API
starts it the same way the CLI does.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.agent.researcher as researcher
import app.agent.trading.interface.runner as runner
import app.main as main
from app.agent.trading.domain.budget import RunBudget, RunTermination
from app.agent.trading.domain.decision_memo import DecisionMemo, EvidenceQuality, Verdict

AS_OF = date(2026, 8, 28)


def _memo() -> DecisionMemo:
    return DecisionMemo(
        ticker="ACN",
        bull_case="STUB",
        bear_case="STUB",
        research_thesis="STUB",
        risk_debate_summary="STUB",
        technical_signal="STUB",
        reasoning="STUB",
        watch_items=[],
        verdict=Verdict.HOLD,
        evidence_quality=EvidenceQuality(
            score=0.0, analyst_coverage=1.0, panel_dispersion=0.0, guard_flags=0
        ),
        data_as_of_date=AS_OF,
    )


class FakeGraph:
    """Stands in for the compiled graph: records every call, runs nothing."""

    def __init__(self, *, values=None, next_=(), result=None):
        self._state = SimpleNamespace(values=values or {}, next=next_)
        self._result = result
        self.invocations: list[tuple[dict | None, dict]] = []

    async def aget_state(self, config):
        return self._state

    async def ainvoke(self, inputs, config):
        self.invocations.append((inputs, config))
        if self._result is not None:
            return self._result
        return {**(inputs or {}), "decision_memo": _memo(), "cost_events": []}


@pytest.fixture
def summaries(monkeypatch):
    logged = []
    monkeypatch.setattr(runner, "log_run_summary", lambda **kw: logged.append(kw))
    return logged


# ---------------------------------------------------------------------------
# runner.start_or_resume
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_a_new_run_starts_with_everything_the_nodes_require(summaries):
    graph = FakeGraph()
    before = datetime.now(timezone.utc)

    outcome = await runner.start_or_resume(
        graph, "ACN", "trading-ACN", AS_OF, max_usd=0.5, wall_clock_timeout_s=600
    )

    assert outcome.status == "started"
    [(inputs, config)] = graph.invocations
    assert inputs["ticker"] == "ACN"
    assert inputs["as_of_date"] == AS_OF
    assert inputs["run_id"] == "trading-ACN"
    budget = inputs["budget"]
    assert budget.max_usd == 0.5
    assert before + timedelta(seconds=600) <= budget.deadline_utc
    assert budget.deadline_utc <= datetime.now(timezone.utc) + timedelta(seconds=600)
    assert config["recursion_limit"] == runner.RECURSION_LIMIT
    assert config["configurable"] == {"thread_id": "trading-ACN"}
    assert len(summaries) == 1 and summaries[0]["run_id"] == "trading-ACN"


@pytest.mark.anyio
async def test_a_completed_thread_is_replayed_not_rerun(summaries):
    graph = FakeGraph(values={"ticker": "ACN", "decision_memo": _memo()})

    outcome = await runner.start_or_resume(graph, "ACN", "trading-ACN", AS_OF)

    assert outcome.status == "completed"
    assert graph.invocations == []
    assert summaries == []


@pytest.mark.anyio
async def test_a_resume_past_its_deadline_is_refused_before_spending(summaries):
    stale = RunBudget(
        max_usd=runner.DEFAULT_MAX_USD,
        deadline_utc=datetime.now(timezone.utc) - timedelta(hours=17),
    )
    graph = FakeGraph(values={"ticker": "ACN", "budget": stale}, next_=("technical",))

    outcome = await runner.start_or_resume(graph, "ACN", "trading-ACN", AS_OF)

    assert outcome.status == "refused"
    assert "REFUSING TO RESUME" in outcome.refusal
    assert graph.invocations == []


@pytest.mark.anyio
async def test_a_live_resume_continues_from_its_checkpoint_and_logs_its_summary(summaries):
    """A resume only follows an attempt that died mid-run, so no summary
    exists for the run yet. It used to write none either — the crash-and-
    resume case the cost reconciliation exists for left no run_summary."""
    live = RunBudget(
        max_usd=runner.DEFAULT_MAX_USD,
        deadline_utc=datetime.now(timezone.utc) + timedelta(minutes=20),
    )
    checkpoint_as_of = date(2026, 8, 20)
    finished = {
        "ticker": "ACN", "as_of_date": checkpoint_as_of, "run_id": "trading-ACN",
        "budget": live, "cost_events": [], "decision_memo": _memo(),
    }
    graph = FakeGraph(
        values={"ticker": "ACN", "budget": live}, next_=("technical",), result=finished
    )

    outcome = await runner.start_or_resume(graph, "ACN", "trading-ACN", AS_OF)

    assert outcome.status == "resumed"
    [(inputs, _)] = graph.invocations
    assert inputs is None   # continue the checkpoint; never re-seed it
    [summary] = summaries
    assert summary["resumed"] is True
    assert summary["run_id"] == "trading-ACN"
    # The checkpoint's analysis date, not the one this call asked for.
    assert summary["as_of_date"] == checkpoint_as_of


@pytest.mark.anyio
async def test_a_new_run_summary_is_not_marked_resumed(summaries):
    await runner.start_or_resume(FakeGraph(), "ACN", "trading-ACN", AS_OF)
    assert summaries[0]["resumed"] is False


def test_a_subset_run_gets_its_own_default_thread():
    assert runner.default_thread_id("ACN") == "trading-ACN"
    assert runner.default_thread_id("ACN", ["technical", "news"]) == "trading-ACN-news+technical"


# ---------------------------------------------------------------------------
# POST /trading/analyze
# ---------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch, tmp_path, summaries):
    """TestClient WITHOUT the lifespan (no `with`), so no Postgres pool or
    checkpointer is opened; each test puts its own fake graph on app.state."""
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)
    return TestClient(main.app)


def test_the_api_seeds_a_new_run_like_the_cli(client, monkeypatch):
    graph = FakeGraph()
    monkeypatch.setattr(main.app.state, "trading_graph", graph, raising=False)

    resp = client.post("/trading/analyze", json={"ticker": " acn "})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "started"
    assert body["thread_id"] == "trading-ACN"
    assert body["decision_memo"]["verdict"] == "hold"
    [(inputs, config)] = graph.invocations
    assert inputs["ticker"] == "ACN"
    assert inputs["as_of_date"] == date.today()
    assert inputs["run_id"] == "trading-ACN"
    assert inputs["budget"].max_usd == runner.DEFAULT_MAX_USD
    assert config["recursion_limit"] == runner.RECURSION_LIMIT


def test_the_api_passes_as_of_date_and_budget_through(client, monkeypatch):
    graph = FakeGraph()
    monkeypatch.setattr(main.app.state, "trading_graph", graph, raising=False)

    resp = client.post("/trading/analyze", json={
        "ticker": "ACN", "as_of_date": "2026-08-28", "max_usd": 0.4,
        "wall_clock_timeout_s": 900, "thread_id": "trading-ACN-api",
    })

    assert resp.status_code == 200, resp.text
    [(inputs, _)] = graph.invocations
    assert inputs["as_of_date"] == AS_OF
    assert inputs["budget"].max_usd == 0.4
    assert inputs["run_id"] == "trading-ACN-api"


def test_an_aborted_run_reports_why_instead_of_crashing(client, monkeypatch):
    aborted = {
        "ticker": "ACN",
        "run_terminated_by": RunTermination.BUDGET_EXCEEDED,
        "budget": RunBudget(max_usd=0.75, deadline_utc=datetime.now(timezone.utc)),
        "cost_events": [],
    }
    monkeypatch.setattr(main.app.state, "trading_graph", FakeGraph(result=aborted), raising=False)

    resp = client.post("/trading/analyze", json={"ticker": "ACN"})

    assert resp.status_code == 200, resp.text
    assert resp.json()["decision_memo"] is None
    assert resp.json()["run_terminated_by"] == "budget_exceeded"


def test_a_refused_resume_is_a_409_and_spends_nothing(client, monkeypatch):
    stale = RunBudget(max_usd=0.75, deadline_utc=datetime.now(timezone.utc) - timedelta(hours=2))
    graph = FakeGraph(values={"ticker": "ACN", "budget": stale}, next_=("technical",))
    monkeypatch.setattr(main.app.state, "trading_graph", graph, raising=False)

    resp = client.post("/trading/analyze", json={"ticker": "ACN"})

    assert resp.status_code == 409
    assert "REFUSING TO RESUME" in resp.json()["detail"]
    assert graph.invocations == []


@pytest.mark.parametrize("field,value", [("max_usd", 0), ("wall_clock_timeout_s", -5)])
def test_a_non_positive_budget_is_rejected(client, monkeypatch, field, value):
    monkeypatch.setattr(main.app.state, "trading_graph", FakeGraph(), raising=False)
    resp = client.post("/trading/analyze", json={"ticker": "ACN", field: value})
    assert resp.status_code == 422


@pytest.fixture
def anyio_backend():
    return "asyncio"
