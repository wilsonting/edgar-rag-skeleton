"""Localizes the residual Phase 6 verdict split to a layer: does the Risk
Judge alone produce different verdicts on a BYTE-IDENTICAL ledger, or does
the split require the panel/ledger to actually vary?

Both risk_determinism_check.py trials so far (AVGO, ASML) re-run the whole
pipeline (panel + Research Manager + Risk Judge) per sample, so a verdict
split there is consistent with variance at ANY of those three layers.
Direct evidence already rules out "Judge-only": AVGO's two temperature=0
replays produced DIFFERENT ledgers (2 of 6 factors drifted, aggregate mass
50/48 and 27/28) before the Judge ever saw them. This script asks the
complementary question with a controlled experiment instead of inference:
freeze one real ledger + one real Research Manager output, then call ONLY
the Risk Judge multiple times against that frozen input. If the verdict
still moves, the Judge itself is a source of variance (in addition to the
panel). If it holds, the split measured so far is fully explained upstream
and Judge-only sampling would be the wrong layer to spend on.

Cost note: there is no persisted ledger from the earlier AVGO/ASML runs to
reuse for free (risk_determinism_check.py never checkpoints one to disk) —
"freeze a saved ledger" in practice means generating exactly one real
panel run + one real Research Manager call first, then repeating only the
Judge. That first part costs about what one trial already costs elsewhere
in this project (~$0.10-0.15, panel + RM); the N=3 Judge-only repeat on top
is ~$0.025/call ≈ $0.075. Call it ~$0.2 total, not the ~$0.03 that would
apply if a ledger were already sitting on disk.

Run:

    uv run python -m scripts.fixed_ledger_judge_repeat TICKER [--as-of YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date

from anthropic import AsyncAnthropic

from app.agent.trading.application.risk_ledger import build_risk_ledger
from app.agent.trading.application.risk_router import RISK_MAX_TURNS
from app.agent.trading.domain.debate import canonical_claims
from app.agent.trading.domain.risk import PERSONAS
from app.agent.trading.infrastructure.risk_port import run_risk_turn
from app.agent.trading.infrastructure.synthesis_port import run_research_manager, run_risk_judge
from scripts.risk_determinism_check import _ledger_scores, build_fixed_debate_state


async def main(ticker: str, as_of: date, n: int) -> None:
    client = AsyncAnthropic()
    fixed_state = await build_fixed_debate_state(ticker, as_of, client)

    print("\n=== Building the ONE frozen ledger + Research Manager output ===")
    turns = []
    for i in range(RISK_MAX_TURNS):
        persona = PERSONAS[i % len(PERSONAS)]
        turn = await run_risk_turn(
            {**fixed_state, "risk_turns": turns}, persona, i, client=client, temperature=None,
        )
        turns.append(turn)
    ledger = build_risk_ledger(turns)
    claims = canonical_claims(fixed_state["debate_turns"])
    research, _, _ = await run_research_manager(
        fixed_state, claims=claims, client=client, temperature=None,
    )
    print(f"  frozen ledger: {len(ledger)} factors, "
          f"contested={[e.factor_id for e in ledger if e.contested]}")
    print(f"  frozen research thesis: {research.thesis[:100]}...")
    frozen_scores = _ledger_scores(ledger)

    print(f"\n=== Calling the Risk Judge {n} times against the IDENTICAL frozen input ===")
    verdicts = []
    for i in range(n):
        payload, cost, _ = await run_risk_judge(
            fixed_state, ledger=ledger, claims=claims, research=research,
            client=client, temperature=None,
        )
        verdicts.append(payload.verdict.value)
        cost_str = f"${cost:.4f}" if cost is not None else "unknown"
        print(f"  [judge-{i + 1}] verdict={payload.verdict.value} "
              f"confidence-inputs unchanged (ledger frozen) cost={cost_str}")

    # The ledger object is mutated nowhere in run_risk_judge — assert that
    # rather than trust it, since a silent mutation would invalidate the
    # whole point of this script.
    assert _ledger_scores(ledger) == frozen_scores, (
        "ledger scores changed across Judge calls — the input was not "
        "actually held fixed, this result is not valid"
    )

    print(f"\n=== Result ===")
    print(f"  verdicts: {verdicts}")
    unanimous = len(set(verdicts)) == 1
    print(f"  {'UNANIMOUS' if unanimous else 'SPLIT'} on a byte-identical ledger + research pack")
    if unanimous:
        print("  -> no Judge-level variance observed here; the AVGO/ASML splits "
              "measured so far are explained by panel/ledger variance, not the "
              "Judge. Sample the panel, not just the Judge.")
    else:
        print("  -> the Judge itself moves on identical input. Panel variance is "
              "not the only source — Judge-level sampling is also load-bearing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("-n", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main(args.ticker, args.as_of, args.n))
