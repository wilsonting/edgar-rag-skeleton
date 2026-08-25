"""Termination guard for the three-persona risk panel.

Pure function of state, same discipline as `debate_router.next_debate_step`:
no I/O, no LLM, no clock, so it is exhaustively testable over every reachable
input in milliseconds at zero API cost — see test_risk_router.py's full
cross-product.

Three independent layers stop the panel, same shape as the debate:

  1. `n >= RISK_MAX_TURNS` here                  — normal operation
  2. `recursion_limit` in the CLI invoke config  — a router bug
  3. runtime asserts at node entry               — a wiring bug that
     (see application/risk_nodes.py)                bypassed the router

RISK_MAX_ROUNDS is the ONLY termination lever. There is deliberately no
productivity/convergence check here at all — not even an inert, observational
one. `debate_router`'s module docstring records that Phase 5's version of
this (`UNPRODUCTIVE_STOP`) needed 100% claim reuse in two consecutive turns to
fire and never came close across five full runs; nothing about three risk
personas scoring a Python-assigned slate makes an LLM-judged "have we
converged" check any more trustworthy, and a termination guard that depends
on a second model's output is a guard with a new failure mode, not a
stronger one.
"""

from app.agent.trading.domain.risk import PERSONAS

RISK_MAX_ROUNDS = 2
RISK_MAX_TURNS = len(PERSONAS) * RISK_MAX_ROUNDS   # 6


def next_risk_step(state) -> str:
    """Returns 'neutral' | 'aggressive' | 'conservative' | 'done'.

    Used as the conditional-edge router for all three risk nodes AND for the
    debate->risk entry edge — the entry edge is conditional for the same
    reason `next_debate_step` guards the analyst->debate edge: a plain edge
    into `neutral_turn` would run one turn before any router saw the state,
    so the no-debate case below could never skip the panel.
    """
    turns = state.get("risk_turns") or []
    n = len(turns)

    # The hard cap. First, unconditional, cannot raise — see
    # debate_router.next_debate_step for why ordering here matters even with
    # a single branch below it.
    if n >= RISK_MAX_TURNS:
        return "done"

    # No debate -> no risk panel. Mirrors the debate router's no-evidence
    # guard one cycle up: a risk panel arguing over a debate that never
    # happened (an empty `--only` selection, or a run where every analyst leg
    # was excluded) would be three personas producing a ledger from nothing —
    # exactly the "reads like a debate and is theatre" failure the debate
    # router already refuses to allow at its own entry.
    if n == 0 and not (state.get("debate_turns") or []):
        return "done"

    return PERSONAS[n % len(PERSONAS)]
