"""Phase 9 Gate A — does the retrieval layer actually return filing text for
every watchlist ticker, before any agent spends money asking it to?

Deliberately probes `RetrievalService` directly rather than the agent's
`ask_edgar` tool: the tool goes over HTTP to the FastAPI app and puts an LLM
between the question and the chunks, so a dead-end there is ambiguous
(no corpus? server down? model declined?). Here a zero is a zero.

The `corpus-status` counts alone are not the check — a ticker can carry rows
in `filings` and still return nothing for the questions the fundamentals
checklist actually asks. Coverage is a claim about retrieval, not storage.
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

from app.application.embedding_service import EmbeddingService  # noqa: E402
from app.application.retrieval_service import RetrievalService  # noqa: E402
from app.infrastructure.queries.corpus_status import CorpusStatusQuery  # noqa: E402
from app.infrastructure.repositories.chunk_repo import (  # noqa: E402
    ChunkRepository,
    ChunkSearchFilters,
)
from app.infrastructure.repositories.db import close_pool, init_pool  # noqa: E402

TICKERS = ["AVGO", "ACN", "NFLX", "FIG", "ASML", "MSFT"]

# One probe per question the Phase 2 fundamentals checklist leans on. Kept
# plain-language, not keyword soup: this must fail the way the agent would.
PROBES = [
    "total revenue for the fiscal year",
    "long-term debt and operating lease obligations",
    "stock-based compensation expense",
    "net cash provided by operating activities",
]

K = 5
MIN_HITS = 3  # per probe, for OK


async def main() -> int:
    await init_pool()
    try:
        summary = {r.ticker: r for r in await CorpusStatusQuery().summary()}
        retrieval = RetrievalService(
            embedding_service=EmbeddingService(), chunk_repo=ChunkRepository()
        )

        rows = []
        for ticker in TICKERS:
            counts, latest, best = [], None, 0.0
            for probe in PROBES:
                hits = await retrieval.retrieve(
                    probe, k=K, filters=ChunkSearchFilters(tickers=[ticker])
                )
                counts.append(len(hits))
                for h in hits:
                    best = max(best, h.similarity)
                    d = h.chunk.filed_date
                    if d and (latest is None or d > latest):
                        latest = d

            if max(counts) == 0:
                status = "EMPTY"
            elif min(counts) >= MIN_HITS:
                status = "OK"
            else:
                status = "THIN"

            s = summary.get(ticker)
            rows.append((ticker, counts, latest, best, status, s))

        print(f"{'ticker':7}{'probe hits':>14}{'filings':>9}{'chunks':>8}"
              f"{'top sim':>9}  {'latest filed':13} status")
        for ticker, counts, latest, best, status, s in rows:
            print(
                f"{ticker:7}{str(counts):>14}"
                f"{(s.filings if s else 0):>9}{(s.chunks if s else 0):>8}"
                f"{best:>9.3f}  {str(latest):13} {status}"
            )

        failed = [t for t, _, _, _, st, _ in rows if st == "EMPTY"]
        thin = [t for t, _, _, _, st, _ in rows if st == "THIN"]
        print()
        if thin:
            print(f"THIN (documented condition for a short filing history): {', '.join(thin)}")
        if failed:
            print(f"GATE A FAILS — no retrievable corpus for: {', '.join(failed)}")
            print("Ingest before running the battery: uv run python -m app.cli ingest <TICKER>")
            return 1
        print("GATE A PASSES — every ticker returns filing text.")
        return 0
    finally:
        await close_pool()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
