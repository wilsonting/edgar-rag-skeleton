"""Per-run state is per asyncio task, not per process.

docs/code_review.md, Medium #11: the research tools kept their provenance
record, calc results, session log, delegated usage, ask_edgar counter and
calc cache in module globals, and researcher kept the vault run stamp in
one — safe for the CLI's one run per process, not for the API server, where
/news-assess and /trading/analyze run the agent inside request handlers and
two overlapping requests reset and read each other's state. A wrong
provenance record defeats the calculate guard and the memo verifier with no
error. The state now lives in ContextVars, so each task sees its own.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

import app.agent.researcher as researcher
import app.agent.tools as tools
from app.domain.token_usage import USAGE_HEADER, TokenUsage, encode_usage_header

# Every wait is bounded: a regression here would otherwise deadlock the suite
# (a run that errors before signalling leaves the other waiting forever).
_TIMEOUT = 5


class _Resp:
    def __init__(self, usage: TokenUsage):
        self.headers = {USAGE_HEADER: encode_usage_header([("deepseek-v4-flash", usage)])}


async def _one_run(n: int, step: asyncio.Event, go_on: asyncio.Event) -> dict:
    """What run_agent's tool calls do to the run state, interleaved with a
    second run at the point where the globals used to collide."""
    tools.reset_run_provenance()
    tools.record_tool_output(f"run {n} revenue was {n}1,234 million")
    tools.record_calc_result(float(n))
    tools._state().ask_edgar_calls += n
    tools._record_delegated_usage(_Resp(TokenUsage(input_tokens=100 * n)))
    step.set()
    await asyncio.wait_for(go_on.wait(), _TIMEOUT)   # the other run resets and records in between
    return {
        "corpus": tools.get_provenance_corpus(),
        "calcs": tools.get_calc_results(),
        "ask_edgar_calls": tools._state().ask_edgar_calls,
        "delegated": tools.get_delegated_usage(),
    }


@pytest.mark.anyio
async def test_two_overlapping_runs_keep_separate_tool_state():
    first_ready, second_ready = asyncio.Event(), asyncio.Event()
    first = asyncio.create_task(_one_run(1, first_ready, second_ready))
    try:
        await asyncio.wait_for(first_ready.wait(), _TIMEOUT)
        second = asyncio.create_task(_one_run(2, second_ready, asyncio.Event()))
        await asyncio.wait_for(second_ready.wait(), _TIMEOUT)   # run 2 has reset and recorded
        a = await asyncio.wait_for(first, _TIMEOUT)             # run 1 now reads
        second.cancel()
    finally:
        first.cancel()

    assert a["corpus"] == "run 1 revenue was 11,234 million"
    assert a["calcs"] == [1.0]
    assert a["ask_edgar_calls"] == 1
    assert a["delegated"] == {"deepseek-v4-flash": TokenUsage(input_tokens=100)}


@pytest.mark.anyio
async def test_two_overlapping_vault_runs_file_into_their_own_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)
    entered = asyncio.Event()

    async def run(stamp: datetime, wait_for: asyncio.Event | None):
        with researcher.vault_run(stamp):
            entered.set()
            if wait_for is not None:
                await asyncio.wait_for(wait_for.wait(), _TIMEOUT)
            return researcher._save_output("# memo", "ACN", "decision")

    release = asyncio.Event()
    a = asyncio.create_task(run(datetime(2026, 9, 11, 10, 0, 0), release))
    await asyncio.wait_for(entered.wait(), _TIMEOUT)
    b_path = await asyncio.wait_for(run(datetime(2026, 9, 11, 11, 0, 0), None), _TIMEOUT)
    release.set()
    a_path = await asyncio.wait_for(a, _TIMEOUT)

    assert a_path.parent.name == "2026-0911-100000"
    assert b_path.parent.name == "2026-0911-110000"
    assert researcher._RUN_STAMP.get() is None


@pytest.mark.anyio
async def test_graph_nodes_see_the_run_folder_set_around_the_invocation():
    """LangGraph runs each node in its own task. The stamp set by vault_run
    around graph.ainvoke must reach the nodes that save from inside the
    graph (fundamentals, technical), or the run's artifacts scatter."""

    class S(TypedDict, total=False):
        seen: str

    async def node(state: S) -> dict:
        stamp = researcher._RUN_STAMP.get()
        return {"seen": stamp.isoformat() if stamp else "none"}

    builder = StateGraph(S)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile()

    with researcher.vault_run(datetime(2026, 9, 11, 12, 30, 0)):
        result = await graph.ainvoke({})

    assert result["seen"] == "2026-09-11T12:30:00"


@pytest.fixture
def anyio_backend():
    return "asyncio"
