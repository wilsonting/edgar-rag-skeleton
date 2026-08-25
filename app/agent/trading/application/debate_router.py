"""Termination guard for the bull/bear debate cycle.

A pure function of state — no I/O, no LLM, no clock. That is what makes it
exhaustively testable: `next_debate_step` can be evaluated over every
reachable input in milliseconds at zero API cost, which is a proof of
termination rather than a sample of it. Five clean live runs are consistent
with a 10%% runaway rate at better-than-even odds; this file is where the
guarantee actually lives.

Three independent layers stop the debate, so no single bug runs away:

  1. `n >= MAX_TURNS` here                       — normal operation
  2. `recursion_limit` in the CLI invoke config  — a router bug
  3. runtime asserts at node entry               — a wiring bug that
     (see application/debate_nodes.py)             bypassed the router

MAX_ROUNDS is the ONLY termination lever. There used to be a second one — an
early stop after two consecutive turns with no new claim_id — removed
2026-08-24 because it never fired. Across every debate measured (five full
pipeline runs plus additional technical-only runs, dozens of turns), a
turn's claims were reused at most ~25%; the `all(...) not t.productive`
clause needs 100% reuse in both of two consecutive turns simultaneously, and
nothing observed came close. `MAX_TURNS` was always what actually stopped the
run. A dead branch inside a termination guard is worse than no branch: it
reads as a second safety layer that is not there, and the next person to
raise MAX_ROUNDS believing it will catch a restatement-heavy debate would be
wrong. If that behavior is ever observed for real, reintroduce a ratio-based
version calibrated against the transcript that showed it — not against a
guess. See known-gaps.md.
"""

from app.agent.trading.application.nodes import ANALYST_OUTPUTS

MAX_ROUNDS = 3
MAX_TURNS = 2 * MAX_ROUNDS


def next_debate_step(state) -> str:
    """Returns 'bull' | 'bear' | 'done'.

    Used as the conditional-edge router for BOTH debate nodes and for the
    analyst -> debate entry edge. The entry edge is conditional for a reason:
    a plain edge into bull_turn would execute one turn before any router ever
    saw the state, so the no-evidence case below could not skip the debate.
    """
    turns = state.get("debate_turns") or []
    n = len(turns)

    # The hard cap. First, unconditional, and it cannot raise. Ordering
    # matters even with a single remaining branch below: this check must
    # never be reachable after something that could throw on a malformed
    # turn, or control falls through to the alternation branch and loops.
    if n >= MAX_TURNS:
        return "done"

    # No evidence -> no debate. A `--only` run that excluded every analyst leg
    # would otherwise produce a debate over an empty pack: two models arguing
    # from nothing, which reads like a debate and is theatre. Same principle
    # as the news caveats — absence of evidence is not neutrality.
    if n == 0 and all(state.get(key) is None for key in ANALYST_OUTPUTS.values()):
        return "done"

    return "bull" if n % 2 == 0 else "bear"


def termination_reason(state) -> str:
    """Why the debate stopped. Only meaningful once `next_debate_step` has
    returned 'done' — called by debate_close_node at exactly that point.

    Two reachable outcomes now, matching the two ways `next_debate_step`
    returns 'done'. "unproductive" was a third outcome before 2026-08-24;
    removed alongside the router clause it named, since termination_reason
    is never called except when next_debate_step just returned 'done', and
    that no longer happens for this reason.
    """
    turns = state.get("debate_turns") or []
    if len(turns) >= MAX_TURNS:
        return "round_cap"
    return "no_evidence"
