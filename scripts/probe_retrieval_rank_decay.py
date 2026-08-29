"""Does trimming `ask_edgar`'s k lose anything? Embeddings only, no LLM calls.

Run before changing ASK_EDGAR_K. Replays the real questions the research
agent asked in a battery, retrieves at k=8, and reports two things a raw
cost saving cannot tell you: how fast similarity decays with rank, and how
often the tail ranks are the ONLY source for a section of the filing.

Measured 2026-08-27 over 102 real questions (NFLX/AVGO/ACN):

    rank 1 mean sim 0.6799 ... rank 8 mean sim 0.6451
    rank 1 -> rank 8 decay: 5.1%     rank 5 -> rank 6 drop: 0.5%
    ranks 6-8 add a section_path absent from ranks 1-5: 60/102 (59%)

There is no cliff to cut at. The tail is nearly as relevant as the head,
and on a majority of questions it carries filing sections nothing else
retrieved -- which is exactly what a cross-section forensic checklist is
looking for.
"""
import asyncio, ast, re, statistics
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)
from app.application.embedding_service import EmbeddingService
from app.application.retrieval_service import RetrievalService
from app.infrastructure.repositories.chunk_repo import ChunkRepository, ChunkSearchFilters
from app.infrastructure.repositories.db import init_pool, close_pool

V = Path("/Users/wilsonsmacmini/Documents/Obsidian Vault/EDGAR-MEMO/memos")

def questions(t):
    p = next(V.glob(f"{t}/20260827/*/{t}-fundamental-provenance.md"))
    out = []
    for c in re.findall(r'\[tool call\] ask_edgar\((\{.*?\})\)\n', p.read_text(), re.S):
        try: out.append(ast.literal_eval(c)["question"])
        except Exception: pass
    return out

async def main():
    await init_pool()
    try:
        r = RetrievalService(embedding_service=EmbeddingService(), chunk_repo=ChunkRepository())
        by_rank = {i: [] for i in range(8)}
        uniq_sections = 0; total_q = 0
        for t in ("NFLX", "AVGO", "ACN"):
            for q in questions(t):
                hits = await r.retrieve(q, k=8, filters=ChunkSearchFilters(tickers=[t]))
                if len(hits) < 8: continue
                total_q += 1
                for i, h in enumerate(hits): by_rank[i].append(h.similarity)
                top5 = {tuple(h.chunk.section_path) for h in hits[:5]}
                tail = {tuple(h.chunk.section_path) for h in hits[5:]}
                if tail - top5: uniq_sections += 1
        print(f"{total_q} questions with a full k=8 result set\n")
        print(f"{'rank':>5}{'mean sim':>11}{'median':>10}")
        for i in range(8):
            v = by_rank[i]
            print(f"{i+1:>5}{statistics.mean(v):>11.4f}{statistics.median(v):>10.4f}")
        m5 = statistics.mean(by_rank[4]); m6 = statistics.mean(by_rank[5])
        m1 = statistics.mean(by_rank[0]); m8 = statistics.mean(by_rank[7])
        print(f"\nrank5 -> rank6 drop: {(m5-m6)/m5*100:.1f}%")
        print(f"rank1 -> rank8 decay: {(m1-m8)/m1*100:.1f}%")
        print(f"\nquestions where ranks 6-8 add a section_path not in ranks 1-5: "
              f"{uniq_sections}/{total_q} ({uniq_sections/total_q*100:.0f}%)")
    finally:
        await close_pool()

asyncio.run(main())
