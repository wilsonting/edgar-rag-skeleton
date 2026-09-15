"""How several rankings become one, and which one the eval harness measures.

Two findings, one subject:

  - `retrieve_full` fused sub-query results by taking each chunk's MAX score
    across sub-queries. RRF scores are rank-derived, so that means "the best
    rank this chunk reached in any sub-query" — a chunk every sub-query
    ranked third scored exactly the same as one a single sub-query ranked
    third and the rest missed. Agreement across sub-queries is the reason to
    decompose a question at all, and the max discarded it.

  - The eval harness offered three modes and none of them was
    `retrieve_full`. `POST /ask` calls `retrieve_full`; the harness measured
    hybrid-without-decomposition, decomposition-without-hybrid, and plain
    vector. Every published retrieval number described a path no caller
    takes.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import eval.runner as runner
from app.application.retrieval_service import RetrievalService, _fuse_across_queries
from app.domain.chunk import Chunk
from app.infrastructure.repositories.chunk_repo import RetrievedChunk


def _rc(cid: int, sim: float, vec: float | None = None) -> RetrievedChunk:
    chunk = Chunk(
        id=cid, section_id=1, content="x", chunk_index=cid, token_count=1,
        ticker="ACN", filed_date=date(2025, 10, 10), filing_type="10-K",
        section_path=["Part II", "Item 7"],
    )
    return RetrievedChunk(chunk=chunk, similarity=sim, vector_similarity=vec)


# ---------------------------------------------------------------------------
# Fusion across sub-queries
# ---------------------------------------------------------------------------

def test_a_chunk_two_sub_queries_agree_on_outranks_one_only_a_single_query_found():
    fused = _fuse_across_queries([[_rc(1, 0.030)], [_rc(2, 0.020)], [_rc(2, 0.025)]], k=8)
    assert [(c.chunk.id, round(c.similarity, 6)) for c in fused] == [(2, 0.045), (1, 0.03)]


def test_the_max_merge_would_have_ranked_them_the_other_way():
    """The behaviour being replaced, stated so the change is not silent."""
    per_query = [[_rc(1, 0.030)], [_rc(2, 0.020)], [_rc(2, 0.025)]]
    by_max = sorted(
        {c.chunk.id: c.similarity for q in per_query for c in q}.items(),
        key=lambda kv: kv[1], reverse=True,
    )
    assert by_max[0][0] == 1                       # old: the single-query chunk won
    assert _fuse_across_queries(per_query, k=8)[0].chunk.id == 2   # new: agreement wins


def test_a_chunk_found_once_still_participates():
    fused = _fuse_across_queries([[_rc(1, 0.03)], [_rc(9, 0.001)]], k=8)
    assert {c.chunk.id for c in fused} == {1, 9}


def test_fusion_truncates_to_k():
    fused = _fuse_across_queries([[_rc(i, 1.0 / i) for i in range(1, 20)]], k=5)
    assert len(fused) == 5
    assert [c.chunk.id for c in fused] == [1, 2, 3, 4, 5]


def test_the_cosine_similarity_carried_through_is_the_best_one_the_chunk_got():
    """`similarity` becomes the fused score; vector_similarity must stay a
    real cosine number, because that is what ask_edgar shows the agent."""
    fused = _fuse_across_queries([[_rc(1, 0.01, vec=0.55)], [_rc(1, 0.03, vec=0.71)]], k=8)
    assert fused[0].similarity == pytest.approx(0.04)
    assert fused[0].vector_similarity == 0.71


def test_no_results_fuse_to_nothing():
    assert _fuse_across_queries([[], []], k=8) == []


# ---------------------------------------------------------------------------
# The dead knob, and the dead method
# ---------------------------------------------------------------------------

def test_the_service_no_longer_takes_a_switch_it_never_read():
    with pytest.raises(TypeError):
        RetrievalService(embedding_service=None, chunk_repo=None, use_hybrid=True)


def test_the_vector_only_decomposition_path_is_gone():
    """Its only caller was the eval harness, measuring a path /ask does not
    take. Keeping it would keep that measurement available."""
    assert not hasattr(RetrievalService, "retrieve_with_decomposition")


# ---------------------------------------------------------------------------
# What the harness measures
# ---------------------------------------------------------------------------

class _Cache:
    def __init__(self, *a, **kw):
        pass

    async def get_or_embed(self, embedder, question):
        return [0.0]

    def flush(self):
        pass


@pytest.fixture
def harness(monkeypatch, tmp_path):
    called = {}

    class _Retrieval:
        def __init__(self, embedder, repo, decomposer=None):
            called["decomposer_built"] = decomposer is not None

        async def retrieve_full(self, q, k=10):
            called["method"] = "retrieve_full"
            from app.application.query_decomposer import DecompositionResult
            return [_rc(1, 0.03)], DecompositionResult(
                original_query=q, was_decomposed=True, sub_queries=[q, "b"]
            )

        async def retrieve_hybrid(self, q, k=10):
            called["method"] = "retrieve_hybrid"
            return [_rc(1, 0.03)]

        async def retrieve_by_embedding(self, vec, k=10):
            called["method"] = "retrieve_by_embedding"
            return [_rc(1, 0.03)]

    monkeypatch.setattr(runner, "RetrievalService", _Retrieval)
    monkeypatch.setattr(runner, "EmbeddingService", lambda *a, **k: object())
    monkeypatch.setattr(runner, "ChunkRepository", lambda *a, **k: object())
    monkeypatch.setattr(runner, "QueryDecomposer", lambda *a, **k: object())
    monkeypatch.setattr(runner, "QuestionEmbeddingCache", _Cache)

    test_set = tmp_path / "test_set.yaml"
    test_set.write_text(
        "questions:\n"
        "  - id: q1\n"
        "    category: synthesis\n"
        "    question: what did cash flow do\n"
        "    components:\n"
        "      - name: cash flow statement\n"
        "        chunk_ids: [1]\n"
    )
    return called, test_set


@pytest.mark.anyio
async def test_the_default_mode_is_the_one_ask_actually_uses(harness):
    called, test_set = harness
    await runner.run_eval(test_set)
    assert called["method"] == "retrieve_full"
    assert runner.DEFAULT_MODE == "full"


@pytest.mark.anyio
@pytest.mark.parametrize("mode,expected", [
    ("full", "retrieve_full"),
    ("hybrid", "retrieve_hybrid"),
    ("vector", "retrieve_by_embedding"),
])
async def test_each_mode_calls_the_path_it_names(harness, mode, expected):
    called, test_set = harness
    await runner.run_eval(test_set, mode=mode)
    assert called["method"] == expected


@pytest.mark.anyio
async def test_only_the_full_mode_pays_for_a_decomposer(harness):
    """The decomposer is an LLM call per question."""
    called, test_set = harness
    await runner.run_eval(test_set, mode="hybrid")
    assert called["decomposer_built"] is False
    await runner.run_eval(test_set, mode="full")
    assert called["decomposer_built"] is True


@pytest.mark.anyio
async def test_an_unknown_mode_lists_the_real_ones(harness):
    _, test_set = harness
    with pytest.raises(ValueError) as exc:
        await runner.run_eval(test_set, mode="decompose")
    assert "full" in str(exc.value) and "POST /ask" in str(exc.value)


@pytest.mark.anyio
async def test_the_decomposition_is_reported_in_the_result(harness):
    _, test_set = harness
    results = await runner.run_eval(test_set, mode="full")
    assert results[0].was_decomposed is True
    assert results[0].sub_queries == ["what did cash flow do", "b"]
    assert results[0].recall_at_5 == 1.0
