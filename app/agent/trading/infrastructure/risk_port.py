"""The LLM side of the three-persona risk panel: evidence pack, one forced
tool call per turn, the guardrails, and the vault transcript.

Same architectural split as debate_port.py: this module is the port (LLM
call, prompts, evidence pack, guardrails, vault I/O); risk_nodes.py is the
application layer (cycle bookkeeping, the layer-3 asserts, the delta shape).

The one structural difference from Phase 5, and the reason domain/risk.py
exists at all: `RiskFactor.factor_id` is Python-assigned, never model-
authored. Phase 5 measured 145 claims across five full debates with 145
distinct `claim_id`s — the model was never once asked to reuse an id and
never did it unprompted (docs/phase6-gate-a-findings.md). A ledger where
three personas score the SAME factor across multiple rounds needs the
opposite of that, so here the model proposes factor TEXT only and Python
assigns the id that everything downstream keys on.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Literal

from app.infrastructure.llm import LLMClient, get_client
from app.infrastructure.llm.models import model_for, warn_if_unpriced
from pydantic import ValidationError

from app.agent.researcher import (
    UsageSummary,
    _save_output,
    log_cost,
)
from app.agent.trading.application.risk_ledger import build_slate, contested_ids
from app.agent.trading.application.risk_router import RISK_MAX_TURNS
from app.agent.trading.domain.debate import DebateTurn, canonical_claims
from app.agent.trading.domain.sanitize import EXTERNAL_TEXT_FRAMING
from app.agent.trading.infrastructure.cost_log import new_event_id, record_cost_event
from app.agent.trading.domain.risk import (
    PERSONAS,
    Persona,
    RiskFactor,
    RiskScore,
    RiskTurn,
    RiskTurnPayload,
)
from app.agent.trading.infrastructure.debate_port import (
    _flag_debate_numbers,
    _inline_refs,
    _norm,
    create_with_temperature_fallback,
    reasoning_config,
    render_transcript as render_debate_transcript,
    report_texts,
)

Phase = Literal["enumerate", "score", "adjudicate", "respond"]

# The project-wide model from .env, same override pattern as DEBATE_MODEL.
RISK_MODEL = model_for("risk_panel")

RISK_MAX_TOKENS = 4000

# Whole-panel ceiling. Measured live (MSFT, 2026-08-25, Haiku 4.5, 6-turn
# 2-round panel): $0.0697-$0.0704. Raised with the round count (2 -> 3,
# Phase 6 gap-closure) since a 9-turn panel costs proportionally more; kept
# with real margin above the measured 6-turn figure rather than tight
# against a linear extrapolation.
RISK_BUDGET_USD = 0.35

warn_if_unpriced(RISK_MODEL, "risk", RISK_BUDGET_USD)

# Forced-failure hooks for the resume tests, mirroring DEBATE_CRASH_AT_TURN.
_CRASH_AT = os.getenv("RISK_CRASH_AT_TURN")
_CRASH_WHEN = os.getenv("RISK_CRASH_WHEN", "before")


def _maybe_crash(turn_index: int, when: str) -> None:
    if _CRASH_AT is None or int(_CRASH_AT) != turn_index or _CRASH_WHEN != when:
        return
    print(f"[risk] FORCED CRASH {when} turn {turn_index} (RISK_CRASH_AT_TURN)")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def turn_phase(turn_index: int) -> Phase:
    """Which of the four phases a turn_index falls in. Pure, and the single
    place this mapping is written down — risk_nodes, the prompt builder and
    the guard checks all read through this rather than each re-deriving it
    from turn_index % 3 // whatever, which is exactly the kind of duplicated
    arithmetic that drifts apart under a later edit.

    Generalized over RISK_MAX_ROUNDS rather than hardcoded to 2 rounds: round
    1 is always enumerate (neutral) + score (aggressive, conservative); every
    round after that is adjudicate (neutral, re-adjudicating whatever is
    STILL contested after the previous round) + respond (aggressive,
    conservative) — so a 3rd round is a second full adjudicate/respond cycle
    over the ledger's current contested set, not a new phase name."""
    if turn_index == 0:
        return "enumerate"
    if turn_index in (1, 2):
        return "score"
    position_in_round = turn_index % len(PERSONAS)   # 0=neutral, 1=aggressive, 2=conservative
    return "adjudicate" if position_in_round == 0 else "respond"


# ---------------------------------------------------------------------------
# Prompts — symmetric by construction across aggressive/conservative, same
# discipline as debate_port's BULL_STANCE/BEAR_STANCE.
# ---------------------------------------------------------------------------

NEUTRAL_STANCE = """\
You are the NEUTRAL moderator of a three-person risk panel. You do not argue
a position. Your job is to enumerate candidate risks on your opening turn,
and to adjudicate disagreements between the other two panelists later."""

AGGRESSIVE_STANCE = """\
You are the AGGRESSIVE panelist: relative to your co-panelist, you weight
upside and discount tail risk more heavily. Argue your scores honestly from
the evidence — 'aggressive' describes your risk tolerance, not a license to
understate a real risk."""

CONSERVATIVE_STANCE = """\
You are the CONSERVATIVE panelist: relative to your co-panelist, you weight
downside and tail risk more heavily. Argue your scores honestly from the
evidence — 'conservative' describes your risk tolerance, not a license to
overstate an unlikely one."""

_STANCE_BY_PERSONA = {
    "neutral": NEUTRAL_STANCE,
    "aggressive": AGGRESSIVE_STANCE,
    "conservative": CONSERVATIVE_STANCE,
}

_PHASE_INSTRUCTIONS: dict[Phase, str] = {
    "enumerate": """\
THIS TURN: enumerate 3-7 candidate risk factors for this security, drawn from
the evidence pack (the analyst reports and the bull/bear debate below). Do
NOT score anything — `scores` must be empty. Each factor needs a `trigger`:
a falsifiable observable (a price level, a metric threshold, a dated event),
never a sentiment. `accept_condition` is not applicable this turn — send
'none'.""",
    "score": """\
THIS TURN: you will be given a SLATE of factor ids. Submit exactly one
`RiskScore` for EVERY id on the slate — omitting one is a structural gap the
ledger records as `missing_scores`, not a neutral abstention. You may
`propose` AT MOST ONE new factor not already on the slate; if you have
nothing to add, `proposes` must be empty. `accept_condition` is REQUIRED:
name the observable that would move you toward your co-panelist's score.""",
    "adjudicate": """\
THIS TURN: you are the neutral moderator. You will be given the CONTESTED
factor ids — where the two panelists' severity or likelihood diverged by 2
or more. Submit a `RiskScore` for each contested id representing your own
adjudicated read; do not score an uncontested id. `proposes` must be empty —
the slate is closed. `accept_condition` is not applicable — send 'none'.""",
    "respond": """\
THIS TURN: you will be given the (still) CONTESTED factor ids after the
moderator's adjudication. Submit an updated `RiskScore` for each one you
still hold a view on; you do not have to move. `proposes` must be empty —
the slate is closed. `accept_condition` is REQUIRED.""",
}

_SYSTEM_TEMPLATE = """\
You are one of three panelists in a structured risk-assessment panel for an
equity position, run AFTER a bull/bear debate over the same evidence.

{STANCE}

You will be given an EVIDENCE PACK (the analyst reports and the debate
transcript) and the risk panel's transcript so far.

{EXTERNAL_TEXT_FRAMING}

HARD RULES — checked in code after you answer:

1. EVERY figure you write in `argument` or a score's `rationale` must appear
   VERBATIM in the evidence pack. Do not compute, re-derive, or restate a
   number in a different unit. Cite, don't compute.
2. A factor's `trigger` must be a falsifiable OBSERVABLE — a specific price,
   a specific metric threshold, or a dated event. "if sentiment sours" is not
   a trigger; "closes below its 200-day average" is.
3. Each proposed factor carries an `evidence_ref`. If it rests on a report or
   the debate, name that source and quote it: `evidence_quote` must be a
   VERBATIM, CONTIGUOUS span of at most 25 words — never spliced with "...".
   If it is your own reasoning rather than a specific citation, set
   evidence_ref='none' and evidence_quote='none'. NEVER send an empty string
   for any field — where a field does not apply, send the literal 'none'.
4. `factor_id` on a NEW proposal: send the literal string 'unassigned'.
   Python assigns the real id; whatever you send there is discarded.
5. `factor_id` on a SCORE: must be one of the ids you were given on the
   slate (or contested-ids list). A score for an id not on the list is
   dropped and flagged, not silently accepted.

Each turn's user message tells you which phase you're in (enumerate, score,
adjudicate, or respond) and what that phase specifically requires — follow
those instructions for the current turn.

Call `submit_risk_turn` exactly once. Say nothing else."""


def _build_system(persona: Persona) -> str:
    """Deliberately NOT a function of `phase` (cost fix, Phase 8 follow-up):
    the phase text used to be baked in here, which meant round 1
    (enumerate/score) and round 2+ (adjudicate/respond) always produced a
    DIFFERENT cached prefix per persona — a guaranteed cache miss on every
    phase transition, live-measured as ~2 of every 3 rounds paying full
    price instead of reading a 90%-cheaper cache hit. `_PHASE_INSTRUCTIONS`
    now goes in the per-turn user message instead (build_risk_evidence_pack
    already isn't phase-dependent either), so the cached prefix — stance +
    evidence pack — is stable across a persona's ENTIRE panel, not just
    within one phase."""
    return (
        _SYSTEM_TEMPLATE.replace("{STANCE}", _STANCE_BY_PERSONA[persona])
        .replace("{EXTERNAL_TEXT_FRAMING}", EXTERNAL_TEXT_FRAMING)
    )


# ---------------------------------------------------------------------------
# Evidence pack — reports + the debate transcript. Reuses debate_port's own
# report renderer rather than re-implementing it, same "one constant, one
# policy" reasoning debate_port already applies to news filtering.
# ---------------------------------------------------------------------------

def build_risk_evidence_pack(state) -> str:
    from app.agent.trading.application.nodes import ANALYST_OUTPUTS

    texts = report_texts(state)
    order = list(ANALYST_OUTPUTS) + ["sentiment"]
    debate_turns: list[DebateTurn] = state.get("debate_turns") or []
    return (
        f"EVIDENCE PACK — {state['ticker'].upper()}\n\n"
        + "\n\n".join(texts[name] for name in order)
        + "\n\nBULL/BEAR DEBATE:\n"
        + render_debate_transcript(debate_turns)
    )


def _debate_claim_corpus(debate_turns: list[DebateTurn]) -> str:
    """Every canonical claim's text + quote, concatenated — the corpus a
    `evidence_ref='debate'` quote is checked against. `canonical_claims`
    specifically (first occurrence per claim_id), matching the meaning
    Phase 5 fixed for any aggregation keyed on claim_id — see
    domain/debate.py's canonical_claims docstring."""
    claims = canonical_claims(debate_turns)
    return "\n".join(f"{c.text} {c.evidence_quote}" for c in claims.values())


def render_risk_transcript(turns: list[RiskTurn]) -> str:
    if not turns:
        return "RISK PANEL TRANSCRIPT: empty — this is the opening turn."
    lines = ["RISK PANEL TRANSCRIPT SO FAR:"]
    for turn in turns:
        lines.append(
            f"\n[turn {turn.turn_index} · round {turn.round_num} · "
            f"{turn.persona.upper()} · phase={turn_phase(turn.turn_index)}]"
        )
        lines.append(turn.payload.argument)
        for factor in turn.payload.proposes:
            quote = f' "{factor.evidence_quote}"' if factor.evidence_quote else ""
            lines.append(
                f"  + {factor.factor_id} [{factor.evidence_ref}] {factor.text} "
                f"(trigger: {factor.trigger}; horizon: {factor.horizon}){quote}"
            )
        for score in turn.payload.scores:
            lines.append(
                f"  · {score.factor_id}: severity={score.severity} "
                f"likelihood={score.likelihood} — {score.rationale}"
            )
        if turn.payload.accept_condition:
            lines.append(f"  accept_condition: {turn.payload.accept_condition}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

RISK_SUBMIT_TOOL = {
    "name": "submit_risk_turn",
    "description": "Submit this turn's risk-panel contribution. Call exactly once.",
    "strict": True,
    "input_schema": _inline_refs(RiskTurnPayload.model_json_schema()),
}


# ---------------------------------------------------------------------------
# Guardrails — all non-fatal (guard_flags), except the two id-space
# integrity rules enforced by construction in `_assemble` below: the slate
# is the one thing Phase 6 exists to make Python-owned, so proposal-count
# violations are truncated at assembly time rather than merely flagged, the
# same way debate_port's structural checks (check_concession, check_rebuts)
# raise instead of flag. Everything else here — trigger quality, quote
# fidelity, number fabrication, accept_condition presence — is a content
# judgment, not an id-space one, and stays a flag: a false positive there
# should cost a reviewer a glance, not a run.
# ---------------------------------------------------------------------------

_COMPARISON_WORDS = re.compile(
    r"\b(above|below|exceeds?|falls?\s+below|reaches?|drops?\s+below|"
    r"rises?\s+above|closes?\s+(?:above|below)|>=?|<=?)\b",
    re.I,
)
_HAS_DIGIT = re.compile(r"\d")
_HAS_DATE_ISH = re.compile(r"\b(20\d{2}|Q[1-4]|January|February|March|April|May|June|"
                            r"July|August|September|October|November|December)\b", re.I)


def _is_falsifiable_trigger(trigger: str) -> bool:
    return bool(
        _HAS_DIGIT.search(trigger)
        or _HAS_DATE_ISH.search(trigger)
        or _COMPARISON_WORDS.search(trigger)
    )


def check_quotes(factors: list[RiskFactor], texts: dict[str, str], debate_corpus: str) -> list[str]:
    """factor_ids (post-assembly, so real ids) whose evidence_quote is not
    actually in the source it names. Same normalization as debate_port's
    check_quotes — imported `_norm`, not reimplemented."""
    flagged = []
    for factor in factors:
        if factor.evidence_ref == "none" or not factor.evidence_quote:
            continue
        corpus = debate_corpus if factor.evidence_ref == "debate" else texts.get(factor.evidence_ref, "")
        if _norm(factor.evidence_quote) not in _norm(corpus):
            flagged.append(factor.factor_id)
    return flagged


def _check_turn(
    turn: RiskTurn,
    *,
    phase: Phase,
    expected_ids: list[str],
    slate: list[str],
    texts: dict[str, str],
    debate_corpus: str,
    prior_risk_corpus: str,
    truncated: list[str],
) -> tuple[list[str], list[str]]:
    """Returns (guard_flags, unquoted_evidence). `truncated` is the list of
    factor texts already dropped by `_assemble` for an over/late proposal —
    passed in so the flag names what was actually removed rather than
    re-deriving it."""
    flags: list[str] = []
    payload = turn.payload

    if truncated:
        kind = "late_proposal" if phase in ("adjudicate", "respond") else "over_proposal"
        flags.append(f"{kind}: dropped {len(truncated)} factor(s): {', '.join(truncated)}")

    if phase in ("score", "adjudicate", "respond"):
        scored_ids = {s.factor_id for s in payload.scores}
        missing = [i for i in expected_ids if i not in scored_ids]
        if missing:
            flags.append(f"slate_incomplete: missing {missing}")
        unknown = sorted({s.factor_id for s in payload.scores} - set(slate))
        if unknown:
            flags.append(f"unknown_factor_id: {unknown}")

    if phase in ("score", "respond") and not payload.accept_condition:
        flags.append("no_accept_condition")

    for factor in payload.proposes:
        if not _is_falsifiable_trigger(factor.trigger):
            flags.append(f"unfalsifiable_trigger: {factor.factor_id} ({factor.trigger!r})")

    unquoted = check_quotes(payload.proposes, texts, debate_corpus)

    # Found live (MSFT, 2026-08-25): a turn quoting an EARLIER risk turn's
    # own number — RF03's own trigger ("RSI falls below 60"), proposed at
    # turn 0 and shown to every later turn via render_risk_transcript in the
    # prompt — was flagged as unbacked, because the corpus checked here was
    # reports + debate only. The risk panel's own running transcript is
    # exactly as citable as a report or a debate claim once a factor has
    # been proposed or a score has been given a number, so it belongs in the
    # corpus too. Prior turns only (not this one) — a turn cannot use its
    # own assertion to back itself.
    number_corpus = "\n\n".join(texts.values()) + "\n\n" + debate_corpus + "\n\n" + prior_risk_corpus
    scan_text = payload.argument + "\n" + "\n".join(s.rationale for s in payload.scores)
    flags.extend(f"unbacked_number: {n}" for n in _flag_debate_numbers(scan_text, number_corpus))

    return flags, unquoted


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------

async def _submit(
    client: LLMClient, system_blocks: list[dict], messages: list[dict],
    temperature: float | None = None,
):
    return await create_with_temperature_fallback(
        client,
        model=RISK_MODEL,
        max_tokens=RISK_MAX_TOKENS,
        **reasoning_config(RISK_MODEL, temperature),
        system=system_blocks,
        messages=messages,
        tools=[RISK_SUBMIT_TOOL],
        tool_choice={
            "type": "tool",
            "name": "submit_risk_turn",
            "disable_parallel_tool_use": True,
        },
    )


def _accumulate(usage: UsageSummary, raw) -> None:
    usage.input_tokens += raw.input_tokens
    usage.cache_write_tokens += raw.cache_creation_input_tokens or 0
    usage.cache_read_tokens += raw.cache_read_input_tokens or 0
    usage.output_tokens += raw.output_tokens


def _tool_block(response):
    return next((b for b in response.content if b.type == "tool_use"), None)


def _extract(response) -> RiskTurnPayload:
    block = _tool_block(response)
    if block is None:
        raise ValidationError.from_exception_data(
            "RiskTurnPayload", [{"type": "missing", "loc": ("submit_risk_turn",), "input": None}]
        )
    return RiskTurnPayload.model_validate(block.input)


_CORRECTION = (
    "That submission did not validate:\n{error}\n\n"
    "Call submit_risk_turn once more, correcting exactly those fields. "
    "Change nothing else."
)


def _retry_messages(messages: list[dict], response, error: Exception) -> list[dict]:
    turns = list(messages)
    if response.content:
        turns.append({"role": "assistant", "content": response.content})
    correction = _CORRECTION.format(error=error)
    results = [
        {"type": "tool_result", "tool_use_id": block.id, "is_error": True, "content": correction}
        for block in response.content
        if block.type == "tool_use"
    ]
    turns.append({"role": "user", "content": results or correction})
    return turns


def _assert_within_budget(ticker: str, turns: list[RiskTurn], this_turn: float | None) -> None:
    total = sum(t.estimated_cost_usd or 0.0 for t in turns) + (this_turn or 0.0)
    if total > RISK_BUDGET_USD:
        raise AssertionError(
            f"risk panel cost ${total:.4f} for {ticker} exceeds the "
            f"${RISK_BUDGET_USD:.2f} per-panel budget after {len(turns) + 1} "
            f"turn(s) — check RISK_MODEL routing and the evidence pack size "
            f"before rerunning"
        )


_NORMALIZE_NONWORD = re.compile(r"[^\w\s]")
_NORMALIZE_SPACE = re.compile(r"\s+")


def _normalize_factor_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — the same
    concept phrased with different capitalization or a trailing period
    still hashes identically. Not fuzzy: "export-control exposure" and
    "export control risk" are different strings after normalization and
    get different ids, on purpose — see `_content_id`'s docstring."""
    return _NORMALIZE_SPACE.sub(" ", _NORMALIZE_NONWORD.sub("", text.lower())).strip()


def _content_id(text: str, taken: set[str]) -> str:
    """factor_id = content hash of the (normalized) factor text, not a
    position in the enumeration.

    Found live (2026-08-26, code review + a diagnostic run correlated to
    the exact replay pair it was explaining): `f"RF{i:02d}"` bound
    identity to ENUMERATION ORDER, which the model does not hold fixed —
    two temperature=0 replays of the identical prompt produced factor
    lists that differed in count (5 vs 6) AND had the same underlying
    concepts under swapped positional ids (RF01 was "MACD deterioration"
    in one replay, "50-day/200-day crossover" in the other). Every
    downstream comparison (ledger diffs, contested-set stability,
    determinism checks) was silently comparing scores attached to
    DIFFERENT real-world risks under a shared label. Python was assigning
    a position, not an identity — the guide's original framing ("Python
    owns identity") was correct in intent and wrong in implementation.

    A content hash fixes the part that's fixable in code: the SAME
    proposed text gets the SAME id regardless of where in the list it
    landed or which replay produced it. It does NOT fix, and cannot fix,
    the model proposing a genuinely different set of risks between
    replays (the 5-vs-6 case above) — that's enumeration variance, a
    property of the free-text generation itself, not an identity bug. A
    closed taxonomy (Python owns the categories, the model only supplies
    per-ticker materiality and trigger) would close that gap too, at
    higher cost; not attempted here. Near-duplicate wording ("export
    control exposure" vs "export-control risk") still hashes differently
    and is a known, accepted gap of this fix specifically — it makes
    replay diffs interpretable (same content -> visibly same id) more than
    it guarantees convergence on paraphrase.

    Collision handling: 4 hex chars is 65,536 buckets for a slate that
    never exceeds ~13 factors (enumerate cap 7 + up to 1 new per scoring
    turn x 6 scoring turns), so a true hash collision is not the expected
    failure mode — but two DIFFERENT factors normalizing to the exact same
    string would collide by construction, so `taken` is checked and a
    numeric suffix appended rather than trusting the hash space size.
    """
    digest = hashlib.sha1(_normalize_factor_text(text).encode()).hexdigest()[:4].upper()
    candidate = f"RF{digest}"
    suffix = 0
    while candidate in taken:
        suffix += 1
        candidate = f"RF{digest}{suffix}"
    return candidate


def _assemble(
    payload: RiskTurnPayload, turn_index: int, persona: Persona, slate: list[str]
) -> tuple[RiskTurn, list[str]]:
    """Python assigns factor_id for every accepted new proposal, and enforces
    the per-phase proposal cap by truncation rather than merely flagging a
    violation — see the module docstring on why this one rule is structural
    rather than observational. Returns (turn, truncated_texts)."""
    phase = turn_phase(turn_index)
    cap = {"enumerate": len(payload.proposes), "score": 1, "adjudicate": 0, "respond": 0}[phase]

    accepted = payload.proposes[:cap]
    dropped = payload.proposes[cap:]
    taken = set(slate)
    for factor in accepted:
        factor.factor_id = _content_id(factor.text, taken)
        taken.add(factor.factor_id)
    payload.proposes = accepted

    turn = RiskTurn(
        turn_index=turn_index,
        round_num=(turn_index // len(PERSONAS)) + 1,
        persona=persona,
        payload=payload,
        slate_at_entry=list(slate),
    )
    return turn, [f.text for f in dropped]


async def run_risk_turn(
    state, persona: Persona, turn_index: int, client: LLMClient | None = None,
    temperature: float | None = None,
) -> RiskTurn:
    """One turn: build the pack, make one forced tool call, run the guards.

    Mirrors debate_port.run_debate_turn's shape (one retry on a schema
    violation, then raise out of the node so the checkpoint carries the
    conversation to that point and a resume re-attempts this turn).

    `temperature`: None in every production call path (risk_nodes.py never
    passes one) — adaptive thinking stays on, matching every turn measured
    so far. Set explicitly only by the Phase 6 determinism/stability check
    scripts (`scripts/risk_determinism_check.py`), which is also the only
    caller that needs thinking disabled to set it at all — see
    debate_port.reasoning_config.
    """
    _maybe_crash(turn_index, "before")

    ticker = state["ticker"]
    turns: list[RiskTurn] = list(state.get("risk_turns") or [])
    debate_turns: list[DebateTurn] = state.get("debate_turns") or []
    phase = turn_phase(turn_index)
    slate = build_slate(turns)
    expected_ids = contested_ids(turns) if phase in ("adjudicate", "respond") else slate

    texts = report_texts(state)
    debate_corpus = _debate_claim_corpus(debate_turns)
    pack = build_risk_evidence_pack(state)
    client = client or get_client(RISK_MODEL)

    system_blocks = [
        {"type": "text", "text": _build_system(persona)},
        {"type": "text", "text": pack, "cache_control": {"type": "ephemeral"}},
    ]
    id_line = (
        f"SLATE (score every one of these): {slate}"
        if phase == "score"
        else f"CONTESTED ids (score only these): {expected_ids}"
        if phase in ("adjudicate", "respond")
        else "No slate yet — this is the enumeration turn."
    )
    user_text = (
        f"{render_risk_transcript(turns)}\n\n{id_line}\n\n"
        f"You are the {persona.upper()} panelist. This is turn {turn_index} "
        f"(round {(turn_index // len(PERSONAS)) + 1}, phase={phase}). "
        f"{_PHASE_INSTRUCTIONS[phase]}\n\n"
        f"Submit your contribution now."
    )
    messages: list[dict] = [{"role": "user", "content": user_text}]

    usage = UsageSummary()
    response = await _submit(client, system_blocks, messages, temperature)
    _accumulate(usage, response.usage)
    try:
        payload = _extract(response)
    except ValidationError as first:
        block = _tool_block(response)
        print(
            f"[risk] {persona} turn {turn_index}: schema violation, one retry "
            f"— stop_reason={response.stop_reason} "
            f"keys={sorted(block.input) if block else None} "
            f"— {'; '.join(str(first).splitlines()[1:5])}"
        )
        messages = _retry_messages(messages, response, first)
        retry = await _submit(client, system_blocks, messages, temperature)
        _accumulate(usage, retry.usage)
        payload = _extract(retry)

    turn, truncated = _assemble(payload, turn_index, persona, slate)

    node_name = f"{persona}_turn"
    event_id = new_event_id(node_name, turn_index=turn_index)
    cost = log_cost(
        ticker, f"trading-risk-{persona}-r{(turn_index // len(PERSONAS)) + 1}", usage,
        model=RISK_MODEL, run_id=state.get("run_id"), event_id=event_id,
    )
    _assert_within_budget(ticker, turns, cost)

    guard_flags, unquoted = _check_turn(
        turn,
        phase=phase,
        expected_ids=expected_ids,
        slate=build_slate(turns) + [f.factor_id for f in turn.payload.proposes],
        texts=texts,
        debate_corpus=debate_corpus,
        prior_risk_corpus=render_risk_transcript(turns),
        truncated=truncated,
    )
    turn.guard_flags = guard_flags
    turn.unquoted_evidence = unquoted
    turn.input_tokens = usage.input_tokens
    turn.output_tokens = usage.output_tokens
    turn.estimated_cost_usd = cost
    turn.cost_event = record_cost_event(event_id, node_name, usage, RISK_MODEL, cost)

    _maybe_crash(turn_index, "after")
    return turn


# ---------------------------------------------------------------------------
# Vault
# ---------------------------------------------------------------------------

def _format_risk_markdown(ticker: str, turns: list[RiskTurn], terminated_by: str) -> str:
    total = sum(t.estimated_cost_usd or 0.0 for t in turns)
    flagged = [f for t in turns for f in t.guard_flags]
    unquoted = [c for t in turns for c in t.unquoted_evidence]

    from app.agent.trading.application.risk_ledger import build_risk_ledger

    ledger = build_risk_ledger(turns)
    contested = [e for e in ledger if e.contested]

    lines = [
        f"# {ticker} — Risk Panel",
        f"**Turns:** {len(turns)} ({len(turns) // len(PERSONAS)} full round(s))",
        f"**Terminated by:** {terminated_by or 'not recorded'}",
        f"**Model:** {RISK_MODEL}",
        "",
    ]

    caveats = []
    if not turns:
        caveats.append(
            "**No risk panel ran.** No debate preceded it, so there was nothing to argue over."
        )
    if terminated_by == "round_cap":
        caveats.append(f"**Reached the {len(turns) // len(PERSONAS)}-round cap.**")
    if flagged:
        caveats.append(f"**{len(flagged)} guard flag(s)** across the panel: {', '.join(flagged[:10])}.")
    if unquoted:
        caveats.append(f"**{len(unquoted)} unverified quote(s):** {', '.join(unquoted[:10])}.")
    if caveats:
        lines += ["## Caveats", ""] + [f"- {c}" for c in caveats] + [""]

    lines += [
        "## Risk ledger",
        "",
        "| Factor | Text | Trigger | Horizon | Proposed by | Neutral | Aggressive | Conservative | Contested |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in ledger:
        def _cell(p: Persona) -> str:
            if p in e.scores:
                sev, lik = e.scores[p]
                return f"S{sev}/L{lik}"
            return "—"
        lines.append(
            f"| `{e.factor_id}` | {e.text} | {e.trigger} | {e.horizon} | {e.proposed_by} "
            f"| {_cell('neutral')} | {_cell('aggressive')} | {_cell('conservative')} "
            f"| {'yes' if e.contested else 'no'} |"
        )

    lines += [
        "",
        "## Summary",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Turns | {len(turns)} |",
        f"| Factors | {len(ledger)} |",
        f"| Contested factors | {len(contested)} |",
        f"| Guard flags | {len(flagged)} |",
        f"| Unverified quotes | {len(unquoted)} |",
        f"| Estimated cost | ${total:.4f} |",
        "",
        "## Transcript",
        "",
    ]

    for turn in turns:
        lines += [
            f"### Turn {turn.turn_index} — {turn.persona.upper()} "
            f"(round {turn.round_num}, phase={turn_phase(turn.turn_index)})",
            "",
            turn.payload.argument,
            "",
        ]
        if turn.payload.proposes:
            lines.append("**Proposed:**")
            for f in turn.payload.proposes:
                lines.append(f"- `{f.factor_id}` [{f.evidence_ref}] {f.text} (trigger: {f.trigger})")
            lines.append("")
        if turn.payload.scores:
            lines.append("**Scores:**")
            for s in turn.payload.scores:
                lines.append(f"- `{s.factor_id}`: severity={s.severity} likelihood={s.likelihood} — {s.rationale}")
            lines.append("")
        if turn.payload.accept_condition:
            lines.append(f"*Accept condition:* {turn.payload.accept_condition}")
        if turn.guard_flags:
            lines.append(f"*Guard flags:* {', '.join(turn.guard_flags)}")
        if turn.unquoted_evidence:
            lines.append(f"*Unverified quotes:* {', '.join(turn.unquoted_evidence)}")
        lines.append("")

    return "\n".join(lines)


def save_risk_transcript(
    ticker: str, turns: list[RiskTurn], terminated_by: str, provenance: str | None = None
) -> Path:
    content = _format_risk_markdown(ticker.upper(), turns, terminated_by)
    total = sum(t.estimated_cost_usd or 0.0 for t in turns)
    return _save_output(
        content, ticker.upper(), "risk", cost_usd=total if turns else None,
        provenance=provenance, model=RISK_MODEL,
    )
