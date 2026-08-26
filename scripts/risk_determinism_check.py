"""Phase 6 exit-criteria verification (the criteria as actually specified,
not the engineering-quality bar the rest of Phase 6 was originally built
against):

  1. DETERMINISM — replay the SAME debate transcript through the risk panel
     + Research Manager + Risk Judge TWICE at temperature=0. The Risk
     Judge's verdict (buy/sell/hold) must be identical both times.
  2. STABILITY — run the same pipeline 3 times at PRODUCTION temperature
     (i.e. the real default: no explicit temperature, adaptive thinking on,
     exactly what every live run in this project actually uses) over the
     same fixed debate transcript. All three verdicts must agree on
     DIRECTION, even though wording will differ.

"Same debate transcript" means the debate itself is generated ONCE (real
API calls, not a hand-built fixture) and then held fixed across every
trial — only the risk panel, Research Manager, and Risk Judge are re-run
per trial.

CODE REVIEW FINDING (2026-08-25), incorporated here: `verdict` alone is the
most-collapsed thing this pipeline emits and the weakest detector of
non-determinism — two runs can produce the identical verdict via different
per-factor ledger scores and a different contested set, which is a FAILED
determinism check that a verdict-only assertion would still pass. This
script now compares four observables, not one:

  - verdict (buy/sell/hold)
  - the ledger's per-factor_id scores, `{factor_id: {persona: (severity,
    likelihood)}}` — exact equality required for the determinism trials
  - the contested set (factor_ids where contested=True)
  - the resolved reference set actually cited in the final memo's prose

For determinism (temperature=0, replayed twice), ALL FOUR must match
exactly, and each is reported PASS/FAIL independently — a verdict match
with a ledger-score or contested-set mismatch is reported as a determinism
FAILURE, not papered over. For stability (production temperature, 3
samples), verdict direction is still the pass/fail bar (the criterion as
specified only asks for that), but confidence spread and the contested-set
Jaccard similarity across the three samples are reported alongside it,
since — per the same review — that is where instability actually shows up
even when the verdict itself doesn't move.

Also worth stating plainly, not just here but every time this script's
output is read: check `--verdict-distribution` first. If every decision
memo this project has ever produced is `hold` (it is, as of this writing —
see trading-agent-known-gaps.md), a stability PASS has limited power to
detect non-determinism in the DIRECTION dimension specifically, because the
sampling distribution may be degenerate rather than genuinely stable. This
script does not manufacture a non-hold input; that requires a different
ticker/date or a deliberately adversarial fixture, tracked as a follow-up
in trading-agent-known-gaps.md rather than solved here.

Cost: one real debate (6 turns) generated once and reused as fixed input;
5 trials of the risk panel (9 turns each, RISK_MAX_ROUNDS=3) + Research
Manager + Risk Judge (2 trials at temperature=0, 3 at production
temperature). Measured (Haiku 4.5 throughout): ~$0.63 for all 5 trials.

Run:

    uv run python -m scripts.risk_determinism_check TICKER [--as-of YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date

from anthropic import AsyncAnthropic

from app.agent.trading.application import nodes
from app.agent.trading.application.debate_router import MAX_TURNS as DEBATE_MAX_TURNS
from app.agent.trading.application.risk_ledger import build_risk_ledger
from app.agent.trading.application.risk_router import RISK_MAX_TURNS
from app.agent.trading.application.technical_indicators import compute_indicators
from app.agent.trading.domain.risk import PERSONAS
from app.agent.trading.domain.technical_report import TechnicalReport
from app.agent.trading.infrastructure.debate_port import run_debate_turn
from app.agent.trading.infrastructure.price_data_port import get_price_history
from app.agent.trading.infrastructure.risk_port import run_risk_turn
from app.agent.trading.infrastructure.synthesis_port import extract_refs, run_synthesis
from app.agent.trading.infrastructure.technical_interpreter_port import interpret_indicators


async def build_fixed_debate_state(ticker: str, as_of: date, client: AsyncAnthropic) -> dict:
    """Real technical report + a real 6-turn debate, generated ONCE. This
    becomes the fixed input every trial below replays against — technical-
    only, same reason every other Phase 6 live check in this project used
    `--only technical`: it needs no local RAG API server and no EDGAR
    corpus, so the cost and moving parts are scoped to what's under test.
    """
    df, source, dropped = await get_price_history(ticker, as_of)
    indicators = compute_indicators(df)
    interpretation, flagged, flagged_claims, _ = await interpret_indicators(ticker, indicators)
    technical_report = TechnicalReport(
        ticker=ticker, as_of_date=df.index[-1].date(), data_source=source,
        bars_used=len(df), bars_dropped_invalid=dropped, indicators=indicators,
        interpretation=interpretation, interpretation_flagged_numbers=flagged,
        interpretation_flagged_claims=flagged_claims,
    )
    state: dict = {"ticker": ticker, "as_of_date": as_of, "technical_report": technical_report}

    turns = []
    for i in range(DEBATE_MAX_TURNS):
        side = "bull" if i % 2 == 0 else "bear"
        turn = await run_debate_turn(state={**state, "debate_turns": turns}, side=side,
                                      turn_index=i, client=client)
        turns.append(turn)
    state["debate_turns"] = turns
    print(f"[fixed input] debate: {len(turns)} turns, "
          f"terminated by round_cap (fixed for every trial below)")
    return state


def _ledger_scores(ledger) -> dict:
    """{factor_id: {persona: [severity, likelihood]}} — a plain, JSON- and
    equality-comparable snapshot of the ledger's actual numbers, not just
    its size. Lists rather than tuples so two independently-built dicts
    compare equal after a JSON round-trip too, not only in-process."""
    return {e.factor_id: {p: list(v) for p, v in e.scores.items()} for e in ledger}


async def run_pipeline_once(
    fixed_state: dict, *, temperature: float | None, client: AsyncAnthropic, label: str,
) -> dict:
    """Fresh risk panel (from an empty risk_turns) + Research Manager + Risk
    Judge, over the SAME fixed debate. Returns a detail dict carrying every
    observable the comparison in `main` needs — verdict alone is not enough,
    see the module docstring."""
    trial_state = dict(fixed_state)
    turns = []
    for i in range(RISK_MAX_TURNS):
        persona = PERSONAS[i % len(PERSONAS)]
        turn = await run_risk_turn(
            {**trial_state, "risk_turns": turns}, persona, i,
            client=client, temperature=temperature,
        )
        turns.append(turn)
    trial_state["risk_turns"] = turns
    trial_state["risk_terminated_by"] = "round_cap"
    trial_state["debate_terminated_by"] = "round_cap"

    # Criterion 2, the direct check rather than the proxy: "every persona
    # scored" (what the ledger's persona keys show) is also true after
    # exactly ONE round — it does not by itself confirm three rounds ran.
    # turn_index/round_num are Python-assigned by construction (never
    # model output), so this is a structural assertion, not a measurement
    # that could vary — but assert it explicitly rather than only arguing
    # it, so a future change to the loop above fails loudly here.
    assert len(turns) == RISK_MAX_TURNS == 9, (
        f"expected exactly 9 risk turns (RISK_MAX_ROUNDS=3), got {len(turns)}"
    )
    assert max(t.round_num for t in turns) == 3, (
        f"expected round_num to reach 3, got max {max(t.round_num for t in turns)}"
    )

    ledger = build_risk_ledger(turns)
    debate_gaps, debate_evidence = nodes._debate_caveats(trial_state)
    risk_gaps, risk_evidence, _ = nodes._risk_caveats(trial_state)

    memo = await run_synthesis(
        trial_state, ledger=ledger,
        base_gaps=debate_gaps + risk_gaps, base_evidence=debate_evidence + risk_evidence,
        as_of=trial_state["as_of_date"], client=client,
        research_temperature=temperature, risk_temperature=temperature,
    )

    contested = sorted(e.factor_id for e in ledger if e.contested)
    resolved_refs = sorted(set(extract_refs(
        memo.bull_case, memo.bear_case, memo.research_thesis,
        memo.risk_debate_summary, memo.reasoning, *memo.watch_items,
    )))
    detail = {
        "label": label,
        "temperature": temperature,
        "verdict": memo.verdict.value,
        "confidence": memo.confidence,
        "ledger_size": len(ledger),
        "ledger_scores": _ledger_scores(ledger),
        "contested": contested,
        "resolved_refs": resolved_refs,
        # The diagnostic a strict per-observable check can't answer on its
        # own (found by review, 2026-08-26): does `factor_id` point at the
        # SAME semantic content across the two replays being compared, or
        # is "RF00" a positional label over an enumeration whose order/
        # membership itself isn't stable? `ledger_scores` matching or not
        # is silent on this — two replays could have identical-looking
        # score matrices while RF00 means something different in each.
        # Captured from turn 0 (the enumeration turn) of THIS SPECIFIC
        # trial, not a separately-sampled probe, so it's the real
        # diagnostic for whatever ledger_scores/contested this detail dict
        # reports, not a proxy for it.
        "turn0_proposes": [
            (f.factor_id, f.text) for f in turns[0].payload.proposes
        ],
        "ledger_factor_text": {e.factor_id: e.text for e in ledger},
    }
    print(f"[{label}] temperature={temperature} verdict={memo.verdict.value} "
          f"ledger={len(ledger)} contested={len(contested)} confidence={memo.confidence}")
    return detail


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _aggregate_stats(detail: dict) -> dict:
    """The restated form of criterion 4: not "is every per-factor score
    identical" (the strict check `_report_determinism` runs, and the one
    that fails), but "do the ledger's AGGREGATE statistics — the thing a
    computed verdict would actually be a function of — hold." Measured on
    AVGO (2026-08-26): contested COUNT and total severity/likelihood mass
    were identical across two temperature=0 replays even though contested
    MEMBERSHIP and individual scores were not. If that holds up as a
    pattern, a verdict computed from these aggregates (rather than emitted
    by the Risk Judge directly) would be the stable quantity criterion 4
    actually needs — this function exists to keep measuring it, not to
    assert the conclusion."""
    scores = detail["ledger_scores"]
    return {
        "contested_count": len(detail["contested"]),
        "severity_mass": sum(v[0] for persona_scores in scores.values() for v in persona_scores.values()),
        "likelihood_mass": sum(v[1] for persona_scores in scores.values() for v in persona_scores.values()),
    }


def _report_aggregate_determinism(results: list[dict]) -> None:
    a, b = results
    agg_a, agg_b = _aggregate_stats(a), _aggregate_stats(b)
    print("\nDETERMINISM — restated (aggregate statistics, not per-factor identity):")
    for key in agg_a:
        match = agg_a[key] == agg_b[key]
        print(f"  {key}: {'MATCH' if match else 'MISMATCH'} ({agg_a[key]} vs {agg_b[key]})")
    overall = agg_a == agg_b
    print(f"  AGGREGATE DETERMINISM: {'PASS' if overall else 'FAIL'} "
          f"— this is NOT the criterion as specified (that's _report_determinism "
          f"above); it's the measurement that decides whether a computed-from-"
          f"aggregates verdict design is viable.")


def _report_identity_diagnostic(results: list[dict]) -> str:
    """The diagnostic requested by review (2026-08-26): distinguishes three
    mechanisms that can each produce "verdict matches, ledger_scores and
    contested_set don't" — and they need different fixes, so which one is
    live matters more than the fact that something diverges.

      A. Slate identity — factor_id is assigned positionally
         (RiskFactor.factor_id = f"RF{i:02d}") over a free-text
         enumeration Python does not control the membership or order of.
         Same id can point at different content across replays, or the
         same content can land under a different id. This is a design
         error, not a sampling artifact, if it's what's actually
         happening.
      B. Prose conditioning — RiskTurnPayload's field order (argument,
         proposes, scores, accept_condition) means autoregressive
         generation samples ~180 words of prose BEFORE the scores, so
         score variance can be downstream of prose variance rather than
         independent of it.
      C. Threshold brittleness — `contested = spread >= 2` turns a ±1
         score drift into a boolean flip whenever a factor sits at
         spread 1 or 2, which is most of them on a 1-5 scale with three
         raters.

    Printed here from the turn-0 proposes lists and ledger factor text of
    the SAME two trials `_report_determinism` already compared — not a
    separately-sampled probe, which would answer a different question.
    """
    a, b = results
    print("\nIDENTITY DIAGNOSTIC — turn-0 proposes, side by side:")
    print(f"  trial 1: {[(fid, text[:60]) for fid, text in a['turn0_proposes']]}")
    print(f"  trial 2: {[(fid, text[:60]) for fid, text in b['turn0_proposes']]}")

    ids_a = [fid for fid, _ in a["turn0_proposes"]]
    ids_b = [fid for fid, _ in b["turn0_proposes"]]
    same_ids_same_order = ids_a == ids_b
    same_id_set = set(ids_a) == set(ids_b)

    text_a, text_b = a["ledger_factor_text"], b["ledger_factor_text"]
    common_ids = set(text_a) & set(text_b)
    same_text_per_id = all(text_a[fid] == text_b[fid] for fid in common_ids)

    print(f"\n  same ids, same order: {same_ids_same_order}")
    print(f"  same id SET (order may differ): {same_id_set}")
    print(f"  for ids present in both ledgers, same text under that id: {same_text_per_id}")

    if not same_id_set or not same_text_per_id:
        mechanism = "A (slate identity) — SEVERE: different factor membership, or same id / different text"
    elif not same_ids_same_order:
        mechanism = "A (slate identity) — same content, different order: same id bound to different text positionally"
    else:
        # Membership and order both hold — check whether the SAME id's
        # scores drifted by >=2 (B: prose conditioning pushed a real score
        # change) or by exactly the amount that crosses the contested
        # threshold without a large underlying move (C: threshold
        # brittleness on a small, real drift).
        max_drift = 0
        threshold_flips = 0
        for fid in common_ids:
            scores_a, scores_b = a["ledger_scores"].get(fid, {}), b["ledger_scores"].get(fid, {})
            for persona in set(scores_a) & set(scores_b):
                sev_a, lik_a = scores_a[persona]
                sev_b, lik_b = scores_b[persona]
                max_drift = max(max_drift, abs(sev_a - sev_b), abs(lik_a - lik_b))
        contested_a, contested_b = set(a["contested"]), set(b["contested"])
        threshold_flips = len(contested_a ^ contested_b)
        if max_drift >= 2:
            mechanism = f"B (prose conditioning) — a same-id score drifted by {max_drift} (>=2)"
        elif threshold_flips:
            mechanism = (
                f"C (threshold brittleness) — scores drift by <=1 per id, but "
                f"{threshold_flips} factor(s) flipped contested status on that drift"
            )
        else:
            mechanism = "none of A/B/C — ids, order, text, and contested status all matched"

    print(f"\n  DIAGNOSIS: {mechanism}")
    return mechanism


def _report_determinism(results: list[dict]) -> bool:
    """Every observable must match exactly across the two temperature=0
    trials. Each dimension is checked and reported independently — a
    verdict match riding on top of a ledger-score or contested-set mismatch
    is a determinism FAILURE, not a pass with an asterisk.

    `checks` names the human-readable dimension; `detail_key` names the
    ACTUAL key in the `run_pipeline_once` detail dict it reads from —
    kept as an explicit mapping (not assumed equal to the check name)
    after a KeyError here (found live, 2026-08-26, AVGO run: "contested_set"
    vs the detail dict's "contested") took down the whole script AFTER the
    determinism section had already found a real mismatch to report,
    losing the stability section for that run entirely. A reporting bug
    that crashes mid-report is worse than a wrong report — it drops
    everything after the crash, not just the one bad line.
    """
    a, b = results
    checks = [
        ("verdict", "verdict", a["verdict"] == b["verdict"]),
        ("ledger_scores", "ledger_scores", a["ledger_scores"] == b["ledger_scores"]),
        ("contested_set", "contested", a["contested"] == b["contested"]),
        ("resolved_refs", "resolved_refs", a["resolved_refs"] == b["resolved_refs"]),
    ]
    print("\nDETERMINISM — per-observable:")
    for name, detail_key, ok in checks:
        print(f"  {name}: {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            print(f"    trial 1: {a[detail_key]}")
            print(f"    trial 2: {b[detail_key]}")
    overall = all(ok for _, _, ok in checks)
    matched = sum(ok for _, _, ok in checks)
    print(f"DETERMINISM: {'PASS' if overall else 'FAIL'} "
          f"({matched}/{len(checks)} observables matched)")
    return overall


def _report_stability(results: list[dict]) -> bool:
    directions = {r["verdict"] for r in results}
    direction_holds = len(directions) == 1

    confidences = [r["confidence"] for r in results]
    confidence_spread = max(confidences) - min(confidences)

    contested_sets = [set(r["contested"]) for r in results]
    pairwise_jaccard = [
        _jaccard(contested_sets[i], contested_sets[j])
        for i in range(len(contested_sets))
        for j in range(i + 1, len(contested_sets))
    ]
    min_jaccard = min(pairwise_jaccard) if pairwise_jaccard else 1.0

    print(f"\nSTABILITY — verdict direction: {'PASS' if direction_holds else 'FAIL'} "
          f"({[r['verdict'] for r in results]})")
    print(f"  confidence spread across samples: {confidence_spread:.2f} "
          f"({confidences})")
    print(f"  contested-set Jaccard similarity (min pairwise): {min_jaccard:.2f} "
          f"— 1.0 means identical contested sets every sample, 0.0 means no overlap")
    print("  (confidence spread and contested-set Jaccard are reported, not gated — "
          "the exit criterion is verdict direction only; a wide spread here with a "
          "held direction means the VERDICT is stable while the RISK READ under it "
          "is not, which the criterion as specified does not catch)")
    return direction_holds


async def main(ticker: str, as_of: date) -> None:
    client = AsyncAnthropic()
    fixed_state = await build_fixed_debate_state(ticker, as_of, client)

    print("\n=== 1. DETERMINISM: same debate, temperature=0, twice ===")
    det_results = []
    for i in range(2):
        detail = await run_pipeline_once(
            fixed_state, temperature=0.0, client=client, label=f"determinism-{i + 1}"
        )
        det_results.append(detail)
    mechanism = _report_identity_diagnostic(det_results)
    determinism_holds = _report_determinism(det_results)
    _report_aggregate_determinism(det_results)

    print("\n=== 2. STABILITY: same debate, production temperature, 3 samples ===")
    stab_results = []
    for i in range(3):
        detail = await run_pipeline_once(
            fixed_state, temperature=None, client=client, label=f"stability-{i + 1}"
        )
        stab_results.append(detail)
    stability_holds = _report_stability(stab_results)

    print("\n=== Summary ===")
    print(json.dumps({"determinism": det_results, "stability": stab_results}, indent=2))
    print(f"\nDeterminism (temp=0, replayed twice, 4 observables): "
          f"{'PASS' if determinism_holds else 'FAIL'}")
    print(f"Stability (production temp, 3 samples, verdict direction): "
          f"{'PASS' if stability_holds else 'FAIL'}")
    print(f"Identity diagnosis: {mechanism}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()
    asyncio.run(main(args.ticker, args.as_of))
