"""Phase 6 domain types for the three-persona risk panel.

Same rule as Phase 4's index-join and Phase 5's debate: the model produces
*content* — factor descriptions, scores, rationale — and Python owns every
id, counter, round number and persona label. Phase 5 measured 145 claims
across five full debates and 145 distinct `claim_id`s: the model was never
once asked to reuse an id and never did it unprompted. The risk panel's whole
value is a ledger where three personas score the SAME factor across two
rounds, which needs the opposite of what Phase 5 observed — so here the slate
is not merely encouraged, it is assigned. See docs/phase6-gate-a-findings.md
for why the Gate A rebuts measurement (100% resolved) does not change this:
`rebuts` is the model pointing at something already in context, `claim_id`
reuse is the model holding a stable handle across turns, and only the second
is what the ledger depends on.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Persona = Literal["neutral", "aggressive", "conservative"]
PERSONAS: tuple[Persona, ...] = ("neutral", "aggressive", "conservative")

Horizon = Literal["days", "weeks", "quarters", "years"]

EvidenceRef = Literal["fundamentals", "technical", "news", "sentiment", "debate", "none"]

# Same fix as domain/debate.py's _BLANK_SENTINELS, same reason: the model
# cannot reliably emit an empty string inside a tool call — asked for one it
# writes a stray "</antml parameter>"-shaped marker instead, which lands in
# whichever field asked for "" and trips whatever guard reads that field as
# "absent". Asking for the literal 'none' avoids it. Duplicated here rather
# than imported from domain.debate because the two domains are otherwise
# independent (same posture as news_digest.py, technical_report.py, etc. each
# owning their own validators) and this is three lines.
_BLANK_SENTINELS = frozenset({"none", "null", "n/a", ""})


def _normalize_blank(value: object) -> object:
    if isinstance(value, str) and value.strip().lower() in _BLANK_SENTINELS:
        return ""
    return value


class RiskFactor(BaseModel):
    """A candidate risk. Proposed by the LLM as text; `factor_id` is
    Python-assigned in `risk_port._assemble` — never trust one on the wire."""

    # default="unassigned", not "" — asking a Claude tool call for a literal
    # empty string is unreliable (it writes a stray parameter-close marker
    # instead; see domain/debate.py's _BLANK_SENTINELS comment for the
    # documented case). Under a strict tool schema this field is REQUIRED on
    # the wire regardless of the default, so the model must send something;
    # `risk_port._assemble` overwrites it unconditionally, so the sent value
    # is never read for real.
    factor_id: str = Field(
        default="unassigned",
        description="Send the literal string 'unassigned'. Python overwrites this field.",
    )
    text: str = Field(description="One sentence naming the risk.")
    trigger: str = Field(
        description=(
            "A falsifiable observable that would confirm this risk is "
            "materialising: a price level, a metric threshold, or a dated "
            "event. NOT a sentiment ('if confidence erodes')."
        )
    )
    horizon: Horizon
    evidence_ref: EvidenceRef = Field(
        description="'none' = reasoning over the debate/reports, not a specific citation."
    )
    evidence_quote: str = Field(
        default="",
        description=(
            "Verbatim span (<=25 words) copied from that report or debate claim. "
            "The literal string 'none' when evidence_ref='none'. Never an empty "
            "string, and never spliced or elided with '...'."
        ),
    )

    _blank = field_validator("evidence_quote", mode="before")(
        lambda v: _normalize_blank(v)
    )


class RiskScore(BaseModel):
    factor_id: str = Field(description="MUST be an id from the slate you were given.")
    # ge/le stay enforced by pydantic on the way back in; the strict tool
    # schema rejects `minimum`/`maximum` outright (see debate_port.py's
    # _STRICT_UNSUPPORTED — found live, a risk turn 400'd on this before the
    # fix), so the 1-5 bound reaches the model only as prose here.
    severity: int = Field(ge=1, le=5, description="Impact if it materialises. 1-5.")
    likelihood: int = Field(ge=1, le=5, description="Probability within the horizon. 1-5.")
    rationale: str = Field(description="<=40 words. Why this score, not the one beside it.")


class RiskTurnPayload(BaseModel):
    """EXACTLY what the LLM returns. No ids it owns, no counters, no persona.

    Field order is deliberate, not alphabetical or "natural": `proposes`
    and `scores` come BEFORE `argument`. Strict structured output fills
    fields in schema-declaration order, autoregressively — with `argument`
    first (as this was originally written), every score is sampled
    CONDITIONED ON ~180 freshly-generated words of prose that themselves
    have no `temperature=0` guarantee. Found live (2026-08-26, code review
    + `scripts/risk_determinism_check.py` on AVGO/ASML): the numeric
    severity/likelihood fields turned out to be reliably reproducible in
    isolation, but not once they follow a large free-text field in the same
    turn — that's a mechanical consequence of autoregressive sampling, not
    a property of the numbers themselves. Moving `argument` last means the
    structured numbers are sampled first and the prose becomes a summary of
    an already-fixed answer, not the thing the answer is conditioned on.
    Nothing is lost for reasoning quality specifically: adaptive thinking
    (see infrastructure/risk_port.py) already runs a private reasoning pass
    before ANY of these fields are generated, `argument` included — it was
    never the model's real chain of thought, just the transcript's.
    """

    proposes: list[RiskFactor] = Field(
        default_factory=list,
        description="Enumeration turn: 3-7. Scoring turns: 0 or 1. Adjudication: 0.",
    )
    scores: list[RiskScore] = Field(default_factory=list)
    accept_condition: str = Field(
        default="",
        description=(
            "What observable would move you toward the opposing persona's "
            "position. Required (non-'none') on scoring turns. The literal "
            "string 'none' on the enumeration and adjudication turns."
        ),
    )
    argument: str = Field(description="<=180 words. Written LAST — see class docstring.")

    _blank = field_validator("accept_condition", mode="before")(
        lambda v: _normalize_blank(v)
    )


class RiskTurn(BaseModel):
    """Payload plus Python-owned metadata. This is what enters TradingState."""

    turn_index: int                  # 0-based, Python
    round_num: int                   # (turn_index // len(PERSONAS)) + 1, Python
    persona: Persona                 # PERSONAS[turn_index % len(PERSONAS)], Python
    payload: RiskTurnPayload
    slate_at_entry: list[str] = Field(default_factory=list)  # ids visible to this turn

    # Per-turn structural findings, same posture as DebateTurn.guard_flags:
    # a cyclic node cannot write to a plain overwrite channel without
    # clobbering its own earlier turns, so findings live on the turn itself
    # and the synthesizer aggregates across turns rather than a state channel
    # accumulating a second copy.
    guard_flags: list[str] = Field(default_factory=list)
    unquoted_evidence: list[str] = Field(default_factory=list)

    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None


class RiskLedgerEntry(BaseModel):
    """Derived, not stored. Built by `build_risk_ledger` from `risk_turns` —
    deliberately absent from TradingState and from ALLOWED_MSGPACK_MODULES.
    A ledger channel would be a second source of truth for content that is
    already a pure function of risk_turns, and (being an overwrite channel
    written near a cycle) exactly the desync shape Phase 5 avoided by
    dropping `debate_summary` in favor of rendering from `debate_turns`."""

    factor_id: str
    text: str
    trigger: str
    horizon: Horizon
    evidence_ref: str
    evidence_quote: str
    proposed_by: Persona
    scores: dict[Persona, tuple[int, int]] = Field(default_factory=dict)
    severity_spread: int = 0        # max - min across present severity scores
    likelihood_spread: int = 0      # max - min across present likelihood scores
    # `contested` (spread >= 2) is a DISPLAY flag only as of the fix below —
    # never feed it into a computation. A hard integer threshold on a 1-5
    # scale with three raters turns a plain ±1 drift into a boolean flip
    # for any factor already sitting at spread 1, which is most of them;
    # §8.3's confidence term used to read that flip as a measurement. Found
    # live (2026-08-26, code review): the same ±1-scale drift that DOESN'T
    # cross this threshold is exactly what temperature=0 non-determinism
    # produces even once the factor-identity bug (see risk_port._assemble)
    # is fixed, so a downstream computation built on this boolean inherits
    # a cliff where the underlying signal is continuous.
    contested: bool = False         # either spread >= 2 — DISPLAY ONLY, see above
    # The continuous replacement: mean of (severity_spread, likelihood_spread)
    # normalized by the maximum possible spread on a 1-5 scale (4), so a
    # single point of drift moves this by at most 0.125 rather than
    # flipping a category. This is what compute_confidence reads.
    normalized_spread: float = 0.0
    missing_scores: list[Persona] = Field(default_factory=list)
