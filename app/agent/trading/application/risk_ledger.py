"""`build_risk_ledger` — pure, no state, no I/O. The Phase 6 deliverable that
is a table, not more prose: one row per `factor_id`, in slate order, keyed
against every persona's score (or its absence).

Contract, the part worth reading twice:

  - Every proposed factor appears exactly once, even if never scored.
  - A persona that omitted a factor appears in `missing_scores`, NOT as a
    default or an average. Silent imputation is how a 1/5 becomes a 3/5 —
    the single most plausible way this function could quietly manufacture a
    number that no persona ever asserted.
  - Spreads are computed over present scores only; a factor with fewer than
    two scores has spread 0 and contested=False, and its missing_scores is
    non-empty — a spread of 0 there does not mean "agreement", it means
    "nothing to compare yet", and callers must read missing_scores to tell
    the two apart.
"""

from __future__ import annotations

from app.agent.trading.domain.risk import PERSONAS, RiskLedgerEntry, RiskTurn

CONTESTED_THRESHOLD = 2  # either spread >= this marks the factor contested (display only)
MAX_SPREAD = 4           # severity/likelihood each range 1-5; widest possible spread


def build_slate(turns: list[RiskTurn]) -> list[str]:
    """factor_ids in the order they were first proposed, transcript order.

    The slate is a pure function of risk_turns rather than a stored channel —
    see domain/risk.py's RiskLedgerEntry docstring for why a second copy of
    this would be a desync surface.
    """
    return [f.factor_id for t in turns for f in t.payload.proposes]


def build_risk_ledger(risk_turns: list[RiskTurn]) -> list[RiskLedgerEntry]:
    entries: dict[str, RiskLedgerEntry] = {}
    order: list[str] = []

    for turn in risk_turns:
        for factor in turn.payload.proposes:
            if factor.factor_id in entries:
                continue  # a duplicate proposal is a guard_flag elsewhere, not a second row
            order.append(factor.factor_id)
            entries[factor.factor_id] = RiskLedgerEntry(
                factor_id=factor.factor_id,
                text=factor.text,
                trigger=factor.trigger,
                horizon=factor.horizon,
                evidence_ref=factor.evidence_ref,
                evidence_quote=factor.evidence_quote,
                proposed_by=turn.persona,
            )

        # `scored_this_turn` is reset every turn, deliberately separate from
        # `entry.scores` (which persists across ALL turns): those two sets
        # answer different questions. A duplicate factor_id in ONE turn's
        # own `scores` list (a model error — the schema doesn't forbid it)
        # should keep the first and drop the second. A LATER turn revising
        # a score it already gave in an EARLIER turn is not a duplicate —
        # it is the "respond"/re-adjudicate phase working as designed
        # (risk_port.py's turn_phase), and must overwrite, not be dropped.
        #
        # Found live (2026-08-25, code review): checking `turn.persona in
        # entry.scores` — persisted state, not turn-local — meant every
        # persona's FIRST score for a factor was permanent. The whole
        # adjudicate/respond cycle (turns 3-5, and now 6-8 for round 3)
        # still ran, still argued, still submitted revised numbers, and the
        # ledger silently kept round 1's numbers forever. `contested` was
        # then computed from stances that were never actually the personas'
        # final position.
        scored_this_turn: set[str] = set()
        for score in turn.payload.scores:
            entry = entries.get(score.factor_id)
            if entry is None:
                continue  # unknown_factor_id — flagged at the turn level, dropped here
            if score.factor_id in scored_this_turn:
                continue  # duplicate score for one factor in ONE turn — first wins
            scored_this_turn.add(score.factor_id)
            entry.scores[turn.persona] = (score.severity, score.likelihood)

    for factor_id in order:
        entry = entries[factor_id]
        entry.missing_scores = [p for p in PERSONAS if p not in entry.scores]
        if len(entry.scores) >= 2:
            severities = [s for s, _ in entry.scores.values()]
            likelihoods = [l for _, l in entry.scores.values()]
            entry.severity_spread = max(severities) - min(severities)
            entry.likelihood_spread = max(likelihoods) - min(likelihoods)
            entry.contested = (
                entry.severity_spread >= CONTESTED_THRESHOLD
                or entry.likelihood_spread >= CONTESTED_THRESHOLD
            )
            # MAX_SPREAD=4: severity/likelihood each range 1-5, so the
            # widest possible disagreement on either axis is 4. Averaging
            # the two axes before normalizing (rather than normalizing each
            # and taking the max) means a factor split on BOTH axes reads
            # as more contested than one split on only one — a real
            # distinction `contested`'s OR-of-two-booleans throws away.
            entry.normalized_spread = (
                (entry.severity_spread + entry.likelihood_spread) / 2
            ) / MAX_SPREAD

    return [entries[fid] for fid in order]


def contested_ids(risk_turns: list[RiskTurn]) -> list[str]:
    """factor_ids the adjudication turn (turn 3) is allowed to touch —
    exactly the ledger's `contested` rows, in slate order. A separate
    entry point rather than making callers filter `build_risk_ledger`'s
    output themselves, since `risk_nodes._risk_turn` needs this before the
    adjudication turn's scores exist and shouldn't have to know the ledger's
    internal field names to get it."""
    return [e.factor_id for e in build_risk_ledger(risk_turns) if e.contested]
