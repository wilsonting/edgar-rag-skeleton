"""Phase 5 domain types for the bull/bear debate.

Same rule as the Phase 4 index-join: the model produces argument *content*;
Python owns every index, counter and side label. If the LLM emitted its own
`round_num`, the termination guard would be reading a field the model can
fabricate — and the guard is the only thing standing between a debate and an
unbounded loop.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.agent.trading.domain.budget import CostEvent

Side = Literal["bull", "bear"]
Stance = Literal["hold", "sharpen", "concede"]

# Which analyst report a claim is drawn from. "none" is a first-class value,
# not a fallback: an argument built on top of other claims is legitimate, and
# calling it report-backed when it isn't is the fabrication this enum exists
# to make visible.
EvidenceRef = Literal["fundamentals", "technical", "news", "sentiment", "none"]


# Deliberately flat, and the docstring below is deliberately short: both of
# these are shipped to the model as part of the tool `input_schema`, so
# implementation notes written here become prompt text. Nesting a model
# inside a model inside a model also makes `model_json_schema()` emit deeper
# `$defs`/`$ref` chains, which debate_port has to inline before sending; one
# level keeps that walk trivial.
# The model cannot reliably emit an empty string into a tool call. Asked for
# one it writes a stray "</antml parameter>" marker instead, and that landed
# in `concession_trigger` on 4 of 4 live turns — which then tripped the
# concession guard on a turn that was not conceding anything. Asking for the
# literal 'none' produced a clean first attempt every time.
#
# So the wire protocol uses a sentinel and Python normalizes it back: "" stays
# the internal meaning of "absent", and nothing downstream has to know. A
# genuine quote consisting of the single word "none" is unreachable through
# this, which is a trade worth making.
_BLANK_SENTINELS = frozenset({"none", "null", "n/a", ""})


def _normalize_blank(value: object) -> object:
    if isinstance(value, str) and value.strip().lower() in _BLANK_SENTINELS:
        return ""
    return value


class DebateClaim(BaseModel):
    """One atomic assertion, backed by a report or by other claims."""

    claim_id: str = Field(
        description=(
            "Short stable slug, e.g. 'vmware-amort-rolloff'. Reuse the SAME id "
            "when restating a claim made in an earlier turn."
        )
    )
    text: str = Field(description="One sentence.")
    evidence_ref: EvidenceRef = Field(
        description=(
            "'none' = reasoning over other claims, not a report-backed fact."
        )
    )
    evidence_quote: str = Field(
        default="",
        description=(
            "Verbatim span (<=25 words) copied from that report. The literal "
            "string 'none' when evidence_ref='none'. Never an empty string, "
            "and never spliced or elided with '...'."
        ),
    )

    _blank = field_validator("evidence_quote", mode="before")(
        lambda v: _normalize_blank(v)
    )


class DebateTurnPayload(BaseModel):
    """EXACTLY what the LLM returns. No indices, no counters, no side."""

    stance: Stance
    concession_trigger: str = Field(
        default="",
        description=(
            "The opposing claim_id that moved you. The literal string 'none' "
            "unless stance='concede'. Never an empty string."
        ),
    )
    argument: str = Field(description="<=200 words.")
    # The bound is stated in the description as well as enforced by pydantic:
    # strict tool schemas reject minItems/maxItems, so the model only learns
    # the range from prose. Pydantic still rejects a violation after the
    # fact — the schema is guidance, the model is the validator.
    claims: list[DebateClaim] = Field(
        min_length=1,
        max_length=5,
        description="Between 1 and 5 claims. Never zero, never more than five.",
    )
    rebuts: list[str] = Field(
        default_factory=list,
        description=(
            "Opponent claim_ids you are directly attacking. Empty on turn 0 only."
        ),
    )

    _blank = field_validator("concession_trigger", mode="before")(
        lambda v: _normalize_blank(v)
    )


class DebateTurn(BaseModel):
    """Payload plus Python-owned metadata. This is what enters TradingState."""

    turn_index: int          # 0-based, assigned by Python
    round_num: int           # (turn_index // 2) + 1, assigned by Python
    side: Side               # assigned by Python from which node ran
    payload: DebateTurnPayload

    # Did this turn introduce a claim_id nobody had used yet? OBSERVATIONAL
    # ONLY as of 2026-08-24 — it no longer terminates the debate. The router
    # used to stop after two consecutive False turns (UNPRODUCTIVE_STOP), but
    # across every debate measured, a turn's claims were never more than
    # ~25% reused, so the clause never fired: MAX_TURNS was always what
    # actually stopped the run. A dead branch inside a termination guard is
    # worse than no branch — it reads as a second safety layer that isn't
    # there. See application/debate_router.py and known-gaps.md.
    productive: bool = True

    # claim_ids in THIS turn whose text differs from the first time that id
    # appeared earlier in the transcript. Not itself a violation — a model
    # paraphrasing the same underlying point across turns is expected, and
    # raising on any wording drift would make claim_id reuse impractical in
    # the first place. Flagged so the drift is visible, same posture as
    # guard_flags/unquoted_evidence below. The actual safety mechanism is
    # `canonical_claims`: any downstream aggregation keyed on claim_id should
    # read through that rather than trusting whichever occurrence it happens
    # to see, because two occurrences of one id are not guaranteed to agree.
    # Found live (ACN, technical-only pack, 2026-08-24): `acn-volume-
    # deteriorating` carried "collapsing conviction that exposes recovery
    # moves to reversal risk" in turn 3 and "deteriorating participation that
    # undermines recovery conviction" in turn 5 — same id, two claims.
    claim_text_drift: list[str] = Field(default_factory=list)

    # Per-turn findings live here rather than in a state channel: a cyclic
    # node cannot write to a plain overwrite channel without clobbering its
    # own earlier turns, and adding a second reducer would be a second source
    # of truth for the same content. The synthesizer aggregates across turns.
    guard_flags: list[str] = Field(default_factory=list)
    unquoted_evidence: list[str] = Field(default_factory=list)

    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None
    # Phase 8: the same call's cost, in the shape TradingState.cost_events
    # needs (cache token breakdown, a stable event_id for resume dedup).
    # Duplicates estimated_cost_usd/input_tokens/output_tokens above rather
    # than replacing them — those two fields serve different readers
    # (the CLI's per-turn cost printout vs. the run-level budget ledger)
    # and this codebase does not disturb an established Python-owned field
    # just because a new consumer wants overlapping data.
    cost_event: CostEvent | None = None


def canonical_claims(turns: list[DebateTurn]) -> dict[str, DebateClaim]:
    """One DebateClaim per claim_id — the FIRST occurrence, transcript order.

    This is the safety mechanism `claim_text_drift` only flags. A `claim_id`
    is meant to be a stable handle for "the same assertion, restated" (the
    system prompt tells the model to reuse ids for exactly that reason), but
    nothing enforces that its `text` stays the same across occurrences, and
    live transcripts show it does not always. Any code that aggregates
    claims by id — Phase 6's risk debate is the reason this exists — must
    NOT flatten `claims` across turns and index by id directly: two entries
    sharing an id can carry different assertions, and a naive index silently
    keeps whichever one it read last, merging them into one claim that
    neither side actually made in those exact words.

    Reading through this function instead fixes the meaning of a claim_id to
    whatever it meant when it was FIRST introduced — first occurrence, not
    last, because the id was coined to name that assertion and every later
    reuse is claiming to restate it, not redefine it. `claim_text_drift`
    remains the record of where a later occurrence disagreed, for a reader
    or a future guard to act on; this function just makes sure that
    disagreement never propagates silently into an aggregate.
    """
    canonical: dict[str, DebateClaim] = {}
    for turn in turns:
        for claim in turn.payload.claims:
            canonical.setdefault(claim.claim_id, claim)
    return canonical
