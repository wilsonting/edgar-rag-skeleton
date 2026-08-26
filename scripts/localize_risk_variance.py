"""Localizes the temp=0 non-determinism found in
scripts/risk_determinism_check.py (AVGO, 2026-08-26: ledger_scores and
contested_set diverged between two temperature=0 replays) to one of three
sources, cheapest-to-check first, then walks turn-by-turn to find exactly
where a byte-identical prompt first produces a different output:

  1. MINE, IN PYTHON — a set or dict rendered into a prompt without sorting
     would make the RENDERED PROMPT itself non-reproducible, independent of
     the model. Checked here at ZERO API cost, at every turn walked: build
     that turn's transcript rendering / evidence pack TWICE from the
     identical prior-turns list and diff byte-for-byte. If this ever fails,
     nothing downstream from that turn was ever going to match and the
     model is exonerated for it.

     Weak evidence against category 1 before any check runs, worth stating
     rather than re-arguing per turn: `risk_determinism_check.py` runs both
     temperature=0 trials in the SAME Python process (one `asyncio.run`),
     so even a raw unsorted `set` (subject to PYTHONHASHSEED, not to
     insertion order — dicts keep insertion order regardless) would iterate
     identically for both trials within that one process. It would only
     bite across separate process invocations.

  2. MINE, UPSTREAM — confirm the "replay" actually held technical_report
     and debate_turns fixed rather than re-deriving them per trial, and
     (for turn N>0) that the PRIOR turns fed to both replays are the same
     objects, not independently re-derived. Checked by identity, not
     inspection alone.

  3. THE MODEL'S — if the rendered prompt for a turn is byte-identical
     across two temperature=0 calls and the OUTPUT still differs, that is
     evidence for category 3 AT THAT TURN SPECIFICALLY: temperature=0
     gives greedy decoding, not bitwise reproducibility, on production LLM
     serving generally. Walking turn-by-turn and stopping at the first
     divergence costs at most 2×N calls to find the Nth turn where it
     first appears, instead of paying for two full 9-turn panels (18 calls)
     and then reasoning about which turn was responsible after the fact.

Run:

    uv run python -m scripts.localize_risk_variance AVGO --as-of 2026-08-25 [--max-turns 9]
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date

from anthropic import AsyncAnthropic

from app.agent.trading.application.risk_ledger import build_slate, contested_ids
from app.agent.trading.domain.risk import PERSONAS
from app.agent.trading.infrastructure.risk_port import (
    build_risk_evidence_pack,
    render_risk_transcript,
    run_risk_turn,
    turn_phase,
)
from scripts.risk_determinism_check import build_fixed_debate_state


def _factor_snapshot(turn) -> list[tuple]:
    return [
        (f.factor_id, f.text, f.trigger, f.horizon, f.evidence_ref, f.evidence_quote)
        for f in turn.payload.proposes
    ]


def _score_snapshot(turn) -> list[tuple]:
    # rationale INCLUDED — found live (AVGO, this script's first run): a
    # snapshot of only (factor_id, severity, likelihood) reported turn 1 as
    # "byte-identical" while the model's actual prose (rationale, argument,
    # accept_condition) had already diverged in wording. That's not
    # noise — it fed straight into turn 2's prompt via the transcript
    # rendering, so a snapshot that omits it produces a false "prompt
    # MISMATCH = category 1" reading one turn later, when the real
    # divergence was category 3 all along, just in an unstructured field
    # this snapshot wasn't looking at.
    return [(s.factor_id, s.severity, s.likelihood, s.rationale) for s in turn.payload.scores]


def _turn_snapshot(turn) -> tuple:
    return (
        turn.payload.argument,
        _factor_snapshot(turn),
        _score_snapshot(turn),
        turn.payload.accept_condition,
    )


def _render_user_text(turns: list, turn_index: int) -> str:
    """Re-derives exactly what run_risk_turn builds as the user message for
    a given turn, from a prior-turns list — the part of the prompt that
    DOES depend on risk_turns (build_risk_evidence_pack does not; it's
    turn-invariant, built once from the fixed debate alone)."""
    phase = turn_phase(turn_index)
    slate = build_slate(turns)
    expected_ids = contested_ids(turns) if phase in ("adjudicate", "respond") else slate
    id_line = (
        f"SLATE (score every one of these): {slate}"
        if phase == "score"
        else f"CONTESTED ids (score only these): {expected_ids}"
        if phase in ("adjudicate", "respond")
        else "No slate yet — this is the enumeration turn."
    )
    return f"{render_risk_transcript(turns)}\n\n{id_line}"


async def main(ticker: str, as_of: date, max_turns: int) -> None:
    client = AsyncAnthropic()
    fixed_state = await build_fixed_debate_state(ticker, as_of, client)

    print("\n=== 0. Fixed-transcript check (category 2, once) ===")
    trial_copy = dict(fixed_state)
    same = (
        trial_copy["technical_report"] is fixed_state["technical_report"]
        and trial_copy["debate_turns"] is fixed_state["debate_turns"]
    )
    print(f"technical_report and debate_turns survive a trial-shaped shallow copy as "
          f"the SAME object: {same}. No RAG retrieval in play (technical-only).")

    print("\n=== 1. Evidence pack determinism (zero API cost, turn-invariant) ===")
    pack_a = build_risk_evidence_pack(fixed_state)
    pack_b = build_risk_evidence_pack(fixed_state)
    print(f"PACK: {'byte-identical' if pack_a == pack_b else 'MISMATCH'} ({len(pack_a)} chars)")

    print(f"\n=== 2. Turn-by-turn walk, temperature=0, stopping at first divergence "
          f"(up to {max_turns} turns) ===")
    turns_a: list = []
    turns_b: list = []
    for i in range(max_turns):
        persona = PERSONAS[i % len(PERSONAS)]

        # Category 1 at this turn: the two replays' prior-turn histories
        # are IDENTICAL up to this point (enforced by the loop stopping at
        # the first divergence), so the rendered user text must match if
        # rendering is a pure function of that history.
        user_a = _render_user_text(turns_a, i)
        user_b = _render_user_text(turns_b, i)
        prompt_ok = user_a == user_b
        print(f"\nturn {i} ({persona}, phase={turn_phase(i)}): "
              f"prompt {'byte-identical' if prompt_ok else 'MISMATCH — category 1, stop here'}")
        if not prompt_ok:
            print(f"  replay A user text: {user_a!r}")
            print(f"  replay B user text: {user_b!r}")
            break

        turn_a = await run_risk_turn(
            {**fixed_state, "risk_turns": turns_a}, persona, i, client=client, temperature=0.0
        )
        turn_b = await run_risk_turn(
            {**fixed_state, "risk_turns": turns_b}, persona, i, client=client, temperature=0.0
        )
        turns_a.append(turn_a)
        turns_b.append(turn_b)

        snap_a, snap_b = _turn_snapshot(turn_a), _turn_snapshot(turn_b)
        output_ok = snap_a == snap_b
        print(f"  output: {'byte-identical' if output_ok else 'DIFFERS'}")
        if not output_ok:
            print(f"  replay A: {snap_a}")
            print(f"  replay B: {snap_b}")
            print(f"\nFIRST DIVERGENCE at turn {i} ({persona}, phase={turn_phase(i)}): "
                  f"prompt was byte-identical, output was not. Category 3 (the model's), "
                  f"localized to this specific turn — everything from turn {i+1} onward "
                  f"in the original full run cascades from here, not from a fresh leak "
                  f"at each later turn.")
            return

    print(f"\nNo divergence found through turn {max_turns - 1} — either the "
          f"non-determinism is further out than checked here, or (less likely, given "
          f"the full run measured a mismatch) this replay happened to agree the whole way.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--max-turns", type=int, default=9)
    args = parser.parse_args()
    asyncio.run(main(args.ticker, args.as_of, args.max_turns))
