"""Gate A (Phase 6 plan, §0): probe the bull/bear debate claim graph before
designing the risk panel's factor-slate schema on top of any assumption about
how it behaves.

Two questions, both empirical:

  1. `distinct_ids / claims` — does the model reuse `claim_id`s across turns,
     or does every turn mint fresh ones? Phase 5's own measurement (five full
     runs, 30 turns) found 145 claims and 145 distinct ids: 0% reuse. That
     number is what `RiskTurnPayload.scores` being Python-slate-constrained
     (domain/risk.py) is a response to — if claim_id reuse doesn't happen
     even once, nothing suggests three risk personas scoring a shared factor
     list would reuse ids either.

  2. `rebuts_resolved / rebuts_total` — does `rebuts` actually point at real
     claims made by the opponent, or does it dangle? This is the one the
     Phase 6 plan calls load-bearing: a low ratio would mean the debate is
     structurally disconnected (six monologues), and synthesis should not
     index-join on `rebuts` at all.

Run against real checkpoints:

    uv run python -m scripts.probe_debate_graph trading-FIG trading-MSFT

Any thread id present in TRADING_CHECKPOINT_DB_URI works. Threads for the
five termination-criterion runs (AVGO, ACN, FIG, ASML, MSFT) are not all
still resident in the checkpoint DB as of this Phase 6 write-up — the
Postgres instance was reused across intervening test runs. Where a thread is
gone, `trading-agent-known-gaps.md` §5(4) already carries this exact
analysis, computed directly against the saved markdown transcripts (a
stricter form: "does `rebuts` resolve to a claim in the OPPONENT'S
IMMEDIATELY PRECEDING turn", not just "some earlier opposing claim") for all
five tickers — 95/95, 100%. That measurement, not a re-run of this script, is
what Gate A's finding rests on; this script exists so the same check can be
re-run cheaply on live checkpoint data as new debates accumulate, including
inside Phase 6 itself once risk_turns exist.
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter

from app.agent.trading.domain.debate import DebateTurn
from app.agent.trading.infrastructure.checkpointer import build_checkpointer


def probe(debate_turns: list[DebateTurn]) -> dict:
    ids_by_turn = [[c.claim_id for c in t.payload.claims] for t in debate_turns]
    all_ids = {cid for turn in ids_by_turn for cid in turn}

    prior_opposing_ids: set[str] = set()
    prior_ids: set[str] = set()
    resolved, dangling = 0, []
    rebuts_total = 0
    for i, turn in enumerate(debate_turns):
        for r in turn.payload.rebuts:
            rebuts_total += 1
            if r in prior_opposing_ids:
                resolved += 1
            else:
                dangling.append((turn.turn_index, r))
        prior_ids |= set(ids_by_turn[i])
        # "opposing" is everything said by the other side up to and
        # including this turn's own side's prior turns — a rebuttal can
        # legitimately name any earlier opposing claim, not only the one
        # immediately before it (that stricter form is what the known-gaps
        # doc measured separately).
        prior_opposing_ids = {
            cid
            for t2 in debate_turns[: i + 1]
            if t2.side != turn.side
            for cid in [c.claim_id for c in t2.payload.claims]
        }

    concessions = [t.turn_index for t in debate_turns if t.payload.stance == "concede"]

    return {
        "claims": sum(len(x) for x in ids_by_turn),
        "distinct_ids": len(all_ids),
        "rebuts_total": rebuts_total,
        "rebuts_resolved": resolved,
        "rebuts_dangling": dangling,
        "concessions": concessions,
    }


async def _load_debate_turns(thread_id: str) -> list[DebateTurn] | None:
    async with build_checkpointer() as checkpointer:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await checkpointer.aget_tuple(config)
        if snapshot is None:
            return None
        return snapshot.checkpoint.get("channel_values", {}).get("debate_turns")


async def _main(thread_ids: list[str]) -> None:
    totals = Counter()
    for thread_id in thread_ids:
        turns = await _load_debate_turns(thread_id)
        if not turns:
            print(f"{thread_id}: no debate_turns in this checkpoint (skipped)")
            continue
        result = probe(turns)
        print(f"{thread_id}: {result}")
        for key in ("claims", "distinct_ids", "rebuts_total", "rebuts_resolved"):
            totals[key] += result[key]

    if totals["claims"]:
        ratio = totals["rebuts_resolved"] / totals["rebuts_total"] if totals["rebuts_total"] else float("nan")
        print(
            f"\nTOTAL: claims={totals['claims']} distinct_ids={totals['distinct_ids']} "
            f"rebuts_resolved/total={totals['rebuts_resolved']}/{totals['rebuts_total']} "
            f"({ratio:.1%})"
        )


if __name__ == "__main__":
    thread_ids = sys.argv[1:] or ["trading-FIG", "trading-MSFT"]
    asyncio.run(_main(thread_ids))
