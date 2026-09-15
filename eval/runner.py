# eval/runner.py
from datetime import date
import json
import yaml
from dataclasses import dataclass
from pathlib import Path

from app.application.query_decomposer import QueryDecomposer
from app.application.retrieval_service import RetrievalService
from app.application.embedding_service import EmbeddingService
from app.infrastructure.repositories.chunk_repo import ChunkRepository
from eval.question_embedding_cache import QuestionEmbeddingCache

# eval/question_embedding_cache.py
CACHE_PATH = Path("eval/.question_embeddings_cache.json")

@dataclass(frozen=True)
class GoldComponent:
    name: str
    chunk_ids: frozenset[int] #frozenset so the dataclass stays hashable/frozen

@dataclass(frozen=True)
class QuestionResult:
    question_id: str
    category: str
    question: str
    components: list[GoldComponent]
    retrieved_chunks: list[int]   # in rank order
    coverage_at_5: float # for synthesis questions, coverage of gold chunks in retrieved chunks at 5
    coverage_at_10: float # for synthesis questions, coverage of gold chunks in retrieved chunks at 10
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float        # 1/rank of first gold chunk, or 0
    success_at_5: float
    success_at_10: float
    was_decomposed: bool
    sub_queries: list[str]


def _parse_components(q: dict) -> list[GoldComponent]:
    """Parse and validate the components block for one question."""
    components_raw = q.get("components")
    if not isinstance(components_raw, list) or not components_raw:
        raise ValueError(
            f"Question {q['id']}: 'components' must be a non-empty list"
        )
    components = []
    for c in components_raw:
        if "name" not in c or "chunk_ids" not in c:
            raise ValueError(
                f"Question {q['id']}: each component needs 'name' and 'chunk_ids'"
            )
        chunk_ids = c["chunk_ids"]
        if not isinstance(chunk_ids, list) or not all(isinstance(x, int) for x in chunk_ids):
            raise ValueError(
                f"Question {q['id']}, component {c['name']!r}: "
                f"chunk_ids must be a list of integers, got {chunk_ids!r}"
            )
        components.append(GoldComponent(name=c["name"], chunk_ids=frozenset(chunk_ids)))
    return components

def _compute_metrics(
    components: list[GoldComponent],
    retrieved_ids: list[int],
) -> dict:
    """Compute all retrieval metrics for one question."""
    all_gold: set[int] = set().union(*(c.chunk_ids for c in components))

    top5 = set(retrieved_ids[:5])
    top10 = set(retrieved_ids[:10])

    recall_5 = len(top5 & all_gold) / len(all_gold) if all_gold else 0.0
    recall_10 = len(top10 & all_gold) / len(all_gold) if all_gold else 0.0
    success_5 = 1.0 if top5 & all_gold else 0.0
    success_10 = 1.0 if top10 & all_gold else 0.0

    rr = 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in all_gold:
            rr = 1.0 / rank
            break

    def coverage_at(k_: int) -> float:
        hits = sum(
            1 for comp in components
            if any(cid in comp.chunk_ids for cid in retrieved_ids[:k_])
        )
        return hits / len(components)

    return {
        "recall_at_5": recall_5,
        "recall_at_10": recall_10,
        "reciprocal_rank": rr,
        "success_at_5": success_5,
        "success_at_10": success_10,
        "coverage_at_5": coverage_at(5),
        "coverage_at_10": coverage_at(10),
    }

async def get_or_embed(embedder, question: str) -> list[float]:
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    if question in cache:
        return cache[question]
    vec = (await embedder.embed_many([question]))[0]
    cache[question] = vec
    CACHE_PATH.write_text(json.dumps(cache))
    return vec

# What each mode measures, and what in production uses it.
#
# `full` is the default because it is the ONLY one that describes what a
# real question goes through: POST /ask calls retrieve_full — decomposition
# AND hybrid — and nothing in this harness used to exercise it. The harness
# offered three modes: hybrid without decomposition, decomposition without
# hybrid, and plain vector. So every retrieval number this project has
# published describes a path no caller takes. The BM25 change in particular
# was justified by numbers measured without decomposition, and fusion
# behaves differently when its inputs are sub-queries.
MODES = {
    "full": "retrieve_full — decomposition + hybrid. What POST /ask does.",
    "hybrid": "retrieve_hybrid — vector + BM25, no decomposition. What metric extraction does.",
    "vector": "vector search alone, from the question-embedding cache. The zero-spend baseline.",
}
DEFAULT_MODE = "full"


async def run_eval(
    test_set_path: Path,
    k: int = 10,
    mode: str = DEFAULT_MODE,
) -> list[QuestionResult]:
    if mode not in MODES:
        raise ValueError(
            f"unknown eval mode {mode!r}. Choose one of:\n  "
            + "\n  ".join(f"{name}: {what}" for name, what in MODES.items())
        )

    test_set = yaml.safe_load(test_set_path.read_text())
    if not test_set or "questions" not in test_set:
        raise ValueError(f"{test_set_path} must have a top-level 'questions' key")
    questions = test_set["questions"]
    if not questions:
        raise ValueError(f"{test_set_path} has no questions")

    embedder = EmbeddingService()
    # The decomposer costs money per question, so it is built only for the
    # mode that uses it.
    decomposer = QueryDecomposer() if mode == "full" else None
    retrieval = RetrievalService(embedder, ChunkRepository(), decomposer=decomposer)
    cache = QuestionEmbeddingCache(CACHE_PATH)

    results: list[QuestionResult] = []
    try:
        for q in questions:
            components = _parse_components(q)

            if mode == "full":
                retrieved, decomposition = await retrieval.retrieve_full(
                    q["question"], k=k
                )
                was_decomposed = decomposition.was_decomposed
                sub_queries = decomposition.sub_queries
            elif mode == "hybrid":
                retrieved = await retrieval.retrieve_hybrid(q["question"], k=k)
                was_decomposed = False
                sub_queries = [q["question"]]
            else:
                question_vec = await cache.get_or_embed(embedder, q["question"])
                retrieved = await retrieval.retrieve_by_embedding(question_vec, k=k)
                was_decomposed = False
                sub_queries = [q["question"]]

            retrieved_ids = [c.chunk.id for c in retrieved]
            metrics = _compute_metrics(components, retrieved_ids)

            results.append(QuestionResult(
                question_id=q["id"],
                category=q["category"],
                question=q["question"],
                components=components,
                retrieved_chunks=retrieved_ids,
                **metrics,
                was_decomposed=was_decomposed,
                sub_queries=sub_queries
            ))
    finally:
        cache.flush()
    return results

def serialize_result(r: QuestionResult) -> dict:
    """JSON-safe serialization. frozensets and dataclasses don't dump natively."""
    return {
        "question_id": r.question_id,
        "category": r.category,
        "question": r.question,
        "components": [
            {"name": c.name, "chunk_ids": sorted(c.chunk_ids)}
            for c in r.components
        ],
        "retrieved_chunks": r.retrieved_chunks,
        "recall_at_5": r.recall_at_5,
        "recall_at_10": r.recall_at_10,
        "reciprocal_rank": r.reciprocal_rank,
        "success_at_5": r.success_at_5,
        "success_at_10": r.success_at_10,
        "coverage_at_5": r.coverage_at_5,
        "coverage_at_10": r.coverage_at_10,
        "was_decomposed": r.was_decomposed,
        "sub_queries": r.sub_queries
    }
