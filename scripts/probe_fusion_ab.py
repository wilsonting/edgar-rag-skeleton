"""Did summing RRF across sub-queries beat taking the max?

The review replaced `retrieve_full`'s cross-sub-query merge: it used to keep
each chunk's MAX score across sub-queries, and now SUMS them. The argument
was that max discards agreement — a chunk every sub-query ranked third
scored exactly what a chunk one sub-query ranked third and the rest missed.
That is an argument, not a measurement, and a pipeline run cannot settle it.

This settles it, without an LLM and without a hand-labelled gold set, using
KNOWN-ITEM retrieval: take a distinctive sentence out of a real chunk, and
that chunk is by construction the right answer for it. Pair two such
sentences from two different chunks and you have a two-part question whose
correct result set is known exactly — which is the shape decomposition
produces and the only shape where the two merges differ at all.

Note the risk this is testing, not just confirming: for a question whose two
halves have two DIFFERENT answers, summing can promote a chunk both halves
liked a little over a chunk one half liked a lot. That would show up here as
worse coverage. A negative result is a real possible outcome.

Cost: one embedding per sub-query (~$0.00005 for the default run). No
completion calls.

    uv run python scripts/probe_fusion_ab.py [--pairs 40] [--k 8]
"""

from __future__ import annotations

import argparse
import asyncio
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_env

load_env()

from app.application.embedding_service import EmbeddingService          # noqa: E402
from app.application.retrieval_service import RetrievalService, _fuse_across_queries  # noqa: E402
from app.infrastructure.repositories.chunk_repo import ChunkRepository  # noqa: E402
from app.infrastructure.repositories.db import close_pool, get_connection, init_pool  # noqa: E402

_SENTENCE = re.compile(r"(?<=[.;])\s+")


def _probe_sentences(content: str, want: int = 1) -> list[str]:
    """Up to `want` distinctive sentences from one chunk, longest first."""
    found = []
    for raw in _SENTENCE.split(content):
        t = " ".join(raw.split())
        if 80 <= len(t) <= 300 and re.search(r"\d", t) and not t.startswith("|"):
            found.append(t)
    found.sort(key=len, reverse=True)
    return found[:want]


def _probe_sentence(content: str) -> str | None:
    """A sentence distinctive enough that its own chunk is the right answer.

    Long, and carrying a figure — generic boilerplate ("See Note 3.") is
    answered equally well by fifty chunks and measures nothing.
    """
    best = None
    for raw in _SENTENCE.split(content):
        s = " ".join(raw.split())
        if 80 <= len(s) <= 300 and re.search(r"\d", s) and not s.startswith("|"):
            if best is None or len(s) > len(best):
                best = s
    return best


def _max_merge(per_query, k):
    """The merge this replaced, reproduced exactly for comparison."""
    best = {}
    for results in per_query:
        for c in results:
            cid = c.chunk.id
            if cid not in best or c.similarity > best[cid].similarity:
                best[cid] = c
    return sorted(best.values(), key=lambda c: c.similarity, reverse=True)[:k]


async def _sample_same_chunk(limit: int) -> list[tuple[int, str, str]]:
    """Two sentences from ONE chunk: sub-queries that AGREE on their answer.

    This is the shape decomposition actually produces — one question split
    into related parts, which retrieve overlapping chunks. Pairing sentences
    from two unrelated chunks (the other shape below) produces disjoint
    result sets, where summing and maxing are identical by definition.
    """
    async with get_connection() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            SELECT id, content FROM chunks
            WHERE embedding IS NOT NULL AND length(content) BETWEEN 900 AND 6000
            ORDER BY id
            """
        )
        rows = await cur.fetchall()
    out = []
    for r in rows:
        two = _probe_sentences(r["content"], want=2)
        if len(two) == 2:
            out.append((r["id"], two[0], two[1]))
    random.Random(20260913).shuffle(out)
    return out[:limit]


async def _sample(limit: int) -> list[tuple[int, str]]:
    async with get_connection() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            SELECT id, content FROM chunks
            WHERE embedding IS NOT NULL AND length(content) BETWEEN 600 AND 6000
            ORDER BY id
            """
        )
        rows = await cur.fetchall()
    out = []
    for r in rows:
        s = _probe_sentence(r["content"])
        if s:
            out.append((r["id"], s))
    random.Random(20260913).shuffle(out)
    return out[:limit]


def _score(fused, targets, k):
    ids = [c.chunk.id for c in fused][:k]
    hits = sum(1 for t in targets if t in ids)
    rr = 0.0
    for rank, cid in enumerate(ids, start=1):
        if cid in targets:
            rr = 1.0 / rank
            break
    return hits / len(targets), 1.0 if hits == len(targets) else 0.0, rr


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--shape", choices=("agree", "split", "both"), default="both",
                    help="agree: two sentences from ONE chunk (what decomposition "
                         "produces). split: two chunks, two answers.")
    args = ap.parse_args()

    await init_pool()
    try:
        retrieval = RetrievalService(EmbeddingService(), ChunkRepository())

        async def run(label, cases):
            totals = {"sum": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
            differed = overlap = 0
            for targets, q_a, q_b in cases:
                per_query = await asyncio.gather(
                    retrieval.retrieve_hybrid(q_a, k=args.k),
                    retrieval.retrieve_hybrid(q_b, k=args.k),
                )
                ids_a = {c.chunk.id for c in per_query[0]}
                ids_b = {c.chunk.id for c in per_query[1]}
                overlap += len(ids_a & ids_b)
                fused_sum = _fuse_across_queries(per_query, k=args.k)
                fused_max = _max_merge(per_query, args.k)
                if [c.chunk.id for c in fused_sum] != [c.chunk.id for c in fused_max]:
                    differed += 1
                for name, fused in (("sum", fused_sum), ("max", fused_max)):
                    for i, v in enumerate(_score(fused, targets, args.k)):
                        totals[name][i] += v
            n = len(cases) or 1
            print(f"\n=== {label}: {len(cases)} questions, k={args.k} ===")
            print(f"chunks found by BOTH sub-queries: {overlap / n:.2f} per question")
            print(f"rankings that differ between the merges: {differed}/{len(cases)}")
            print(f"{'merge':<6} {'recall@k':>9} {'both@k':>8} {'MRR':>7}")
            for name in ("max", "sum"):
                r, b, m = (v / n for v in totals[name])
                print(f"{name:<6} {r:>9.3f} {b:>8.3f} {m:>7.3f}")
            return totals["sum"][0] / n, totals["max"][0] / n

        if args.shape in ("agree", "both"):
            rows = await _sample_same_chunk(args.pairs)
            cases = [({cid}, a, b) for cid, a, b in rows]
            rs, rm = await run("AGREE - two parts of one question (what decomposition makes)", cases)
            print(f"recall@k: {'sum better' if rs > rm else 'max better' if rm > rs else 'no difference'}"
                  f" ({rs:.3f} vs {rm:.3f})")

        if args.shape in ("split", "both"):
            pool = await _sample(args.pairs * 2)
            pairs = [(pool[i], pool[i + 1]) for i in range(0, len(pool) - 1, 2)][: args.pairs]
            cases = [({a[0], b[0]}, a[1], b[1]) for a, b in pairs]
            rs, rm = await run("SPLIT - two unrelated parts, two answers", cases)
            print(f"recall@k: {'sum better' if rs > rm else 'max better' if rm > rs else 'no difference'}"
                  f" ({rs:.3f} vs {rm:.3f})")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
