"""One turn's tool calls run in waves, not one at a time.

docs/code_review.md, Medium #13: run_agent awaited each tool call in turn.
The calls in one turn are independent (the model issued them together), and
gpt-5.6-luna sent 6-8 ask_edgar calls per turn, each waiting 10-40 s on the
server — wall clock, not cost, bound the run. They now run in waves of
TOOL_CONCURRENCY, with the run budget re-checked before each wave.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import app.agent.researcher as researcher


def _block(i: int, name: str = "ask_edgar"):
    return SimpleNamespace(type="tool_use", id=f"t{i}", name=name, input={"n": i})


@pytest.fixture
def running(monkeypatch):
    """execute_tool that records how many calls are in flight at once."""
    state = {"in_flight": 0, "peak": 0, "order": []}

    async def fake_execute(name, inputs):
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        await asyncio.sleep(0.01)
        state["in_flight"] -= 1
        state["order"].append((name, inputs["n"]))
        return f"{name}:{inputs['n']}"

    monkeypatch.setattr(researcher, "execute_tool", fake_execute)
    return state


@pytest.mark.anyio
async def test_a_turns_calls_overlap_up_to_the_concurrency_limit(running, monkeypatch):
    monkeypatch.setattr(researcher, "TOOL_CONCURRENCY", 4)
    results = await researcher._run_tool_calls([_block(i) for i in range(6)], None, researcher.UsageSummary())

    assert results == [f"ask_edgar:{i}" for i in range(6)]   # call order, always
    assert running["peak"] == 4


@pytest.mark.anyio
async def test_waves_are_faster_than_one_at_a_time(running, monkeypatch):
    blocks = [_block(i) for i in range(8)]
    loop = asyncio.get_running_loop()

    monkeypatch.setattr(researcher, "TOOL_CONCURRENCY", 1)
    t0 = loop.time(); await researcher._run_tool_calls(blocks, None, researcher.UsageSummary())
    serial = loop.time() - t0

    monkeypatch.setattr(researcher, "TOOL_CONCURRENCY", 4)
    t0 = loop.time(); await researcher._run_tool_calls(blocks, None, researcher.UsageSummary())
    waved = loop.time() - t0

    assert waved < serial / 2


@pytest.mark.anyio
async def test_ingest_runs_alone_so_later_calls_read_what_it_wrote(running, monkeypatch):
    monkeypatch.setattr(researcher, "TOOL_CONCURRENCY", 4)
    blocks = [_block(0), _block(1, "ingest_ticker"), _block(2), _block(3)]

    await researcher._run_tool_calls(blocks, None, researcher.UsageSummary())

    order = running["order"]
    assert order.index(("ingest_ticker", 1)) > order.index(("ask_edgar", 0))
    assert order.index(("ingest_ticker", 1)) < min(order.index(("ask_edgar", 2)), order.index(("ask_edgar", 3)))


@pytest.mark.anyio
async def test_the_budget_is_rechecked_between_waves(running, monkeypatch):
    """Overshoot is bounded by one wave: once the check fires, every call not
    yet started is answered with the reason instead."""
    monkeypatch.setattr(researcher, "TOOL_CONCURRENCY", 2)
    stop = lambda usage: "budget_exceeded" if running["order"] else None

    results = await researcher._run_tool_calls([_block(i) for i in range(5)], stop, researcher.UsageSummary())

    assert results[:2] == ["ask_edgar:0", "ask_edgar:1"]
    assert all(r.startswith("RUN BUDGET REACHED: budget_exceeded") for r in results[2:])
    assert len(running["order"]) == 2


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# Sub-query retrieval, and clients built once
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_decomposed_sub_queries_are_retrieved_concurrently_and_fused_by_agreement():
    from app.application.query_decomposer import DecompositionResult
    from app.application.retrieval_service import RetrievalService
    from datetime import date
    from app.domain.chunk import Chunk
    from app.infrastructure.repositories.chunk_repo import RetrievedChunk

    state = {"in_flight": 0, "peak": 0}

    def _rc(cid, sim):
        chunk = Chunk(id=cid, section_id=1, content="x", chunk_index=cid, token_count=1,
                      ticker="ACN", filed_date=date(2025, 10, 10), filing_type="10-K",
                      section_path=["Part II", "Item 7"])
        return RetrievedChunk(chunk=chunk, similarity=sim)

    class Decomposer:
        async def decompose(self, q):
            return DecompositionResult(original_query=q, was_decomposed=True, sub_queries=["a", "b", "c"])

    service = RetrievalService(embedding_service=None, chunk_repo=None, decomposer=Decomposer())

    async def fake_hybrid(q, k=8, filters=None):
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        await asyncio.sleep(0.01)
        state["in_flight"] -= 1
        return {"a": [_rc(1, 0.03), _rc(2, 0.02)], "b": [_rc(2, 0.025)], "c": [_rc(3, 0.01)]}[q]

    service.retrieve_hybrid = fake_hybrid
    chunks, decomposition = await service.retrieve_full("q", k=8)

    assert state["peak"] == 3
    # Chunk 2 was found by TWO sub-queries (0.02 + 0.025) and chunk 1 by one
    # (0.03), so chunk 2 ranks first. Under the previous max-across-queries
    # merge it scored 0.025 and came second — agreement between sub-queries,
    # which is the entire reason to decompose a question, counted for
    # nothing. See retrieval_service._fuse_across_queries.
    assert [(c.chunk.id, round(c.similarity, 6)) for c in chunks] == [
        (2, 0.045), (1, 0.03), (3, 0.01),
    ]
    assert decomposition.sub_queries == ["a", "b", "c"]


def test_endpoints_use_the_shared_clients_when_the_server_built_them(monkeypatch):
    import app.main as main

    shared = object()
    monkeypatch.setattr(main.app.state, "embedder", shared, raising=False)
    assert main._embedder() is shared

    monkeypatch.setattr(main.app.state, "embedder", None, raising=False)
    monkeypatch.setattr(main, "EmbeddingService", lambda: "fresh")
    assert main._embedder() == "fresh"   # no lifespan (tests, or a failed build): per request
