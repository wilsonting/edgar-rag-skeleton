"""The two-call synthesis split: a Research Manager that synthesizes the
bull/bear DEBATE into cases and a thesis, and a Risk Judge that synthesizes
the risk-panel LEDGER and issues the pipeline's ONLY verdict.

This is a Phase 6 gap-closure rewrite. The original single-call
`run_synthesis` (one model producing bull_case/bear_case/risk_narrative/
verdict together) is gone: the actual spec calls for two distinct roles —
"Research Manager (Sonnet) synthesizes the bull/bear debate; Risk Judge
(Sonnet) synthesizes the risk debate" — and a determinism/stability
guarantee on THE RISK VERDICT specifically, which only means something once
there is a Risk Judge call whose own output IS that verdict.

`ResearchManagerPayload` originally also carried a `preliminary_verdict`,
reviewed by the Risk Judge and either affirmed or overridden. Removed
(2026-08-26, code review): it was an intermediate discrete emission nothing
downstream required, shown to the Judge as prior context — an anchoring
effect on the agent that actually decides. Measured live (ASML): the
preliminary verdict alone flip-flopped sell/hold/sell across three
production-temperature samples of the identical fixed debate. The Research
Manager's job is cases, not a verdict.

Model choice diverges from that spec text on purpose: both roles follow the
project-wide `LLM_CLAUDE_MODEL` (Haiku 4.5), not a Sonnet pin. Sonnet 5
turned out to have deprecated `temperature` outright — see
debate_port.create_with_temperature_fallback — which undercuts exactly the
determinism guarantee these two roles exist to support; Haiku 4.5 honors
the parameter for real.

Still the highest-fabrication-risk part of the pipeline — the only calls
that can invent a figure appearing nowhere upstream, and the only output a
human reads directly. Same discipline as before: neither payload carries a
number or a quote. Every narrative sentence cites `[C:claim_id]` (a debate
claim) or `[RFnn]` (a risk-ledger factor); Python resolves those, renders
their evidence, and runs a numeric fabrication guard on the rendered text.
Confidence is computed from observables, never model-emitted.

Imports from debate_port are LOCAL to each function rather than at module
level — see the identical note in the pre-rewrite version of this module
and in risk_port.py: debate_port imports `ANALYST_OUTPUTS` from
application.nodes at its OWN module level, and nodes.py imports from this
module, so a top-level import here completes the cycle.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

from app.infrastructure.llm import LLMClient, get_client
from app.infrastructure.llm.models import model_for, warn_if_unpriced
from pydantic import ValidationError

from app.agent.researcher import (
    UsageSummary,
    log_cost,
)
from app.agent.trading.domain.budget import CostEvent
from app.agent.trading.domain.debate import DebateClaim, DebateTurn, canonical_claims
from app.agent.trading.domain.decision_memo import (
    DecisionMemo,
    ResearchManagerPayload,
    RiskJudgePayload,
    Verdict,
)
from app.agent.trading.domain.risk import RiskLedgerEntry, RiskTurn
from app.agent.trading.domain.sanitize import EXTERNAL_TEXT_FRAMING
from app.agent.trading.infrastructure.cost_log import new_event_id, record_cost_event

# Follows the project-wide model from .env (LLM_CLAUDE_MODEL), same override
# pattern as DEBATE_MODEL/RISK_MODEL — NOT pinned to Sonnet. The spec names
# Sonnet explicitly for these two roles, but Sonnet 5 turned out to have
# deprecated `temperature` outright (create_with_temperature_fallback exists
# because of it), which undercuts exactly the determinism guarantee these
# two calls exist to support; Haiku 4.5 accepts the parameter for real, so
# switching back makes the temperature=0 replay a genuine controlled
# condition for the Research Manager/Risk Judge too, not just the risk
# panel. Still overridable per-role independent of the project-wide default.
RESEARCH_MANAGER_MODEL = model_for("research_manager")
RISK_JUDGE_MODEL = model_for("risk_judge")

SYNTHESIS_MAX_TOKENS = 4000

# Combined ceiling across BOTH calls (Research Manager + Risk Judge).
# Measured live on Sonnet pricing at $0.1232 (Research Manager) + $0.2139
# (Risk Judge) per call — kept as the ceiling even after the Haiku switch
# since it's meant to catch a prompt-bloat regression, not track whichever
# model is cheapest today.
SYNTHESIS_BUDGET_USD = 0.30

for _model in (RESEARCH_MANAGER_MODEL, RISK_JUDGE_MODEL):
    warn_if_unpriced(_model, "synthesis", SYNTHESIS_BUDGET_USD)

_CRASH_AT = os.getenv("SYNTHESIS_CRASH_AT")   # "research" | "risk_judge" | None


def _maybe_crash(when: str) -> None:
    if _CRASH_AT != when:
        return
    print(f"[synthesis] FORCED CRASH at {when} (SYNTHESIS_CRASH_AT)")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


class SynthesisReferenceError(Exception):
    """A reference token names no real claim_id or factor_id, on the SECOND
    attempt (the model was already shown the unresolved ids once and asked
    to correct them). Raised rather than dropped — a silently-dropped
    reference leaves a fluent sentence whose support has vanished.

    `cost_events` (Phase 8, default []): whatever CostEvents were already
    incurred before this raise. Empty when raised from inside
    `_resolve_with_retry` — that path has never logged cost either (a
    pre-existing gap, not one this phase introduces); populated by
    `run_synthesis` when it re-raises a Risk Judge failure, so the Research
    Manager's already-spent cost isn't lost with it.
    """

    def __init__(self, message: str, *, cost_events: list[CostEvent] | None = None):
        super().__init__(message)
        self.cost_events = cost_events or []


class SynthesisFabricationError(Exception):
    """A number in a load-bearing field (Research Manager's `thesis`, Risk
    Judge's `risk_narrative`/`reasoning`) appears in no report, no debate
    claim, and no risk factor. Blocking, not flagging — see
    `_numeric_guard`.

    `cost_events`: see SynthesisReferenceError's docstring — same contract.
    """

    def __init__(self, message: str, *, cost_events: list[CostEvent] | None = None):
        super().__init__(message)
        self.cost_events = cost_events or []


class MemoVerificationError(Exception):
    """The FULLY ASSEMBLED memo failed `verify_decision_memo` even though
    every individual call that produced it passed its own guard at
    generation time. More serious than a plain SynthesisFabricationError/
    SynthesisReferenceError: those catch a bad call before its output is
    used; this catches a bad artifact built from calls that each looked
    clean in isolation — an assembly-step bug (e.g. a future DecisionMemo
    field wired in without going through the guard), not a model output
    the pipeline already knows how to reject."""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Rule 1 in both prompts below already forbids writing ANY number in the
# narrative fields, and both models violate it routinely: NFLX's memo opens
# "free cash flow accelerated 37%, leverage fell 41 basis points gross and
# 34 basis points net, and operating margin expanded 280 bps" — five
# numbers in the first sentence of a field where none are allowed. The
# guard that runs after does not catch this, because it checks whether a
# number is BACKED, not whether it should be there at all.
#
# So this is written as a fallback rule rather than a repetition of rule 1:
# given that figures do get restated, the units have to survive the
# restatement. Ported from the research agent's own prompt
# (app/agent/prompts.py), where the identical rule demonstrably works —
# NFLX's fundamentals report says "a 0.41-turn improvement" and it is the
# SYNTHESIS layer, which had no such rule, that turned it into "41 basis
# points". A 27% reduction in gross leverage reported as 0.41 percentage
# points is the memo's lead positive claim, wrong by two orders of
# magnitude, and every guard in the pipeline passed it.
_UNITS_RULE = """\
UNITS SURVIVE RESTATEMENT. Rule 1 says not to restate figures at all. If
   you do anyway, the unit must be right: basis points and percentage
   points apply ONLY to metrics already expressed as a percentage —
   margins, rates, yields. A leverage, coverage or any ratio expressed in
   "x" (turns) changes in TURNS: 1.50x to 1.09x is a 0.41-turn improvement,
   never "41 basis points". Never attach a currency sign to a ratio
   ("2.80x", not "$2.80x"). Gross and net are different figures — if the
   source distinguishes them, say which one you mean."""


RESEARCH_MANAGER_SYSTEM = f"""\
You are the Research Manager for an equity research pipeline. You will be
given the analyst reports and the bull/bear debate transcript for one
ticker. Your job is to synthesize the DEBATE — you do not see any risk
assessment; that happens after you, in a separate review you have no
visibility into.

{EXTERNAL_TEXT_FRAMING}

HARD RULES — checked in code after you answer:

1. LANGUAGE AND JUDGEMENT ONLY. Never write a number, a dollar figure, a
   percentage, or a direct quote in `bull_case`, `bear_case`, or `thesis`.
   Cite instead: every claim from the debate is `[C:claim_id]`, exactly as
   it appears in the transcript. Python resolves these into the actual
   evidence — you never retype a figure already established elsewhere.
2. Every load-bearing sentence in `thesis` carries at least one citation.
3. You do NOT issue a verdict. Your job ends at synthesizing the debate
   into cases and a thesis; a separate Risk Judge, who also reviews the
   risk panel you never see, decides buy/sell/hold. Do not hedge your
   `thesis` toward a particular direction in anticipation of that decision
   — describe what the debate actually supports.
4. {_UNITS_RULE}

Call `submit_research_synthesis` exactly once. Say nothing else."""

RISK_JUDGE_SYSTEM = f"""\
You are the Risk Judge for an equity research pipeline — the sole decision
maker. You will be given the analyst reports, the bull/bear debate, the
three-persona risk panel's ledger, AND the Research Manager's own synthesis
of the debate (their bull_case, bear_case, and thesis — they do not issue a
verdict; that is your job alone, informed by risk factors they never saw).

{EXTERNAL_TEXT_FRAMING}

HARD RULES — checked in code after you answer:

1. LANGUAGE AND JUDGEMENT ONLY. Never write a number, a dollar figure, a
   percentage, or a direct quote in `risk_narrative`, `reasoning`, or
   `watch_items`. Cite instead: every risk factor is `[RFnn]` exactly as
   shown in the ledger; you may also cite `[C:claim_id]` when your reasoning
   needs to reference the underlying debate. Python resolves these into the
   actual evidence — you never retype a figure already established
   elsewhere.
2. `reasoning` must state which risk factor(s) were decisive for `verdict`
   and why — not whether you agree with the Research Manager, since they
   offered no verdict to agree or disagree with. Every load-bearing
   sentence carries a citation.
3. `verdict` is buy/sell/hold — no fourth option, no hedge. This is the
   PIPELINE'S ONLY verdict.
4. `watch_items`: at most 5, each an observable (not a vague sentiment) that
   would change this read, each citing the `[RFnn]` factor it comes from.
5. {_UNITS_RULE}

A risk ledger with no rows, or a report/debate marked "NOT RUN"/"empty", is
missing evidence, not neutral evidence — do not infer anything from its
absence.

Call `submit_risk_judgment` exactly once. Say nothing else."""


# ---------------------------------------------------------------------------
# Evidence packs
# ---------------------------------------------------------------------------

def build_research_pack(state) -> str:
    """Reports + debate transcript ONLY — the Research Manager never sees
    the risk ledger, by design (it synthesizes the debate; risk review is a
    separate, later judgment)."""
    from app.agent.trading.infrastructure.debate_port import (
        build_evidence_pack,
        render_transcript,
    )

    debate_turns: list[DebateTurn] = state.get("debate_turns") or []
    return (
        build_evidence_pack(state)
        + "\n\nBULL/BEAR DEBATE (cite claims exactly as shown, e.g. [C:claim-id]):\n"
        + render_transcript(debate_turns)
    )


def _render_ledger(ledger: list[RiskLedgerEntry]) -> str:
    if not ledger:
        return "RISK LEDGER: none — no risk panel ran."
    lines = ["RISK LEDGER (cite factor ids exactly as shown, e.g. [RF1A2B]):"]
    for e in ledger:
        scores = ", ".join(
            f"{p}=severity{sev}/likelihood{lik}" for p, (sev, lik) in e.scores.items()
        ) or "no scores yet"
        quote = f' "{e.evidence_quote}"' if e.evidence_quote else ""
        lines.append(
            f"[{e.factor_id}] {e.text} (trigger: {e.trigger}; horizon: {e.horizon}; "
            f"evidence: {e.evidence_ref}{quote}) — {scores}"
            + (" — CONTESTED" if e.contested else "")
            + (f" — missing: {', '.join(e.missing_scores)}" if e.missing_scores else "")
        )
    return "\n".join(lines)


def _render_research_output(research: ResearchManagerPayload) -> str:
    return (
        "RESEARCH MANAGER'S SYNTHESIS OF THE DEBATE (context for your own "
        "verdict — they issued none; the decision is yours alone):\n"
        f"Bull case: {research.bull_case}\n"
        f"Bear case: {research.bear_case}\n"
        f"Thesis: {research.thesis}"
    )


def build_risk_judge_pack(
    state, ledger: list[RiskLedgerEntry], research: ResearchManagerPayload
) -> str:
    from app.agent.trading.infrastructure.debate_port import (
        build_evidence_pack,
        render_transcript,
    )

    debate_turns: list[DebateTurn] = state.get("debate_turns") or []
    return (
        build_evidence_pack(state)
        + "\n\nBULL/BEAR DEBATE:\n"
        + render_transcript(debate_turns)
        + "\n\n"
        + _render_ledger(ledger)
        + "\n\n"
        + _render_research_output(research)
    )


# ---------------------------------------------------------------------------
# Reference resolution — generalized over a list of texts rather than one
# payload type, since Research Manager and Risk Judge have different field
# shapes and this logic is otherwise identical for both.
# ---------------------------------------------------------------------------

# `RF\d+` was written when `factor_id` was `f"RF{i:02d}"`. `_content_id`
# (risk_port.py) replaced that with a 4-char SHA-1 prefix in hex, uppercase,
# with an optional numeric collision suffix — so a real id is `RF487E` far
# more often than `RF3755`, and only the ~15% of ids that happen to be
# all-digits ever matched. Found live (2026-08-26, Phase 9 Gate work) on a
# real NFLX memo: five factors cited, four of them hex, and the rendered
# Evidence section listed exactly one — the all-digit `RF3755`. The other
# four were invisible to BOTH `_render_evidence` (so the reader gets a
# citation with nothing behind it) and `resolve_refs` (so a hallucinated
# factor id containing any of A-F passed post-hoc verification silently).
#
# Deliberately `RF[0-9A-Za-z]+` rather than a pattern that re-encodes the
# current id shape: matching is only about FINDING citation-shaped tokens,
# and `resolve_refs`/`_render_evidence` both decide validity by membership
# in the real slate anyway. A looser matcher hands an unknown id to that
# membership check — which reports it — where a tighter one drops it on the
# floor. This bug was the tight version failing exactly that way, and the
# whole test suite missed it because every fixture still used `RF00`.
_REF_PATTERN = re.compile(r"\[C:([\w.-]+)\]|\[(RF[0-9A-Za-z]+)\]")


def extract_refs(*texts: str) -> list[str]:
    """Every `[C:id]`/`[RFnn]` token across the given fields, IN ORDER,
    duplicates included."""
    joined = "\n".join(texts)
    return [m.group(1) or m.group(2) for m in _REF_PATTERN.finditer(joined)]


def resolve_refs(
    texts: list[str], claims: dict[str, DebateClaim], ledger_by_id: dict[str, RiskLedgerEntry]
) -> list[str]:
    known = set(claims) | set(ledger_by_id)
    return sorted({r for r in extract_refs(*texts) if r not in known})


def _render_evidence(
    texts: list[str], claims: dict[str, DebateClaim], ledger_by_id: dict[str, RiskLedgerEntry]
) -> list[str]:
    seen: set[str] = set()
    lines: list[str] = []
    for ref in extract_refs(*texts):
        if ref in seen:
            continue
        seen.add(ref)
        if ref in claims:
            c = claims[ref]
            quote = f' "{c.evidence_quote}"' if c.evidence_quote else ""
            lines.append(f"[C:{ref}] ({c.evidence_ref}) {c.text}{quote}")
        elif ref in ledger_by_id:
            e = ledger_by_id[ref]
            quote = f' "{e.evidence_quote}"' if e.evidence_quote else ""
            lines.append(f"[{ref}] {e.text} (trigger: {e.trigger}){quote}")
    return lines


# ---------------------------------------------------------------------------
# Confidence — computed from observables, never model-emitted
# ---------------------------------------------------------------------------

def compute_confidence(
    state, ledger: list[RiskLedgerEntry], debate_turns: list[DebateTurn]
) -> float:
    """A judgement call in its weights, not a calibration. Unchanged by the
    two-call split — it reads final state (coverage, contestation, guard
    flags), not which call produced which field. Calibrating the weights
    against realised outcomes remains a later-phase item, not attempted
    here.

    Reads `normalized_spread` (continuous, 0-1), NOT `contested` (boolean).
    Found live (2026-08-26, code review): `contested` is a hard threshold
    (spread >= 2) on a 1-5 scale with three raters, so a plain ±1 score
    drift — exactly what temperature=0 non-determinism produces even with
    factor identity fixed — flips the boolean for any factor already
    sitting at spread 1. Feeding that flip into confidence meant a 1-point
    drift could move confidence by 0.3 (a third of the whole contestation
    term) or by nothing, depending on which side of the cliff it landed.
    `normalized_spread`'s average moves confidence by at most ~0.03 per
    point of drift on a typical ledger — continuous input, continuous
    output, matching the reasoning `contested_share` was supposed to
    capture without the cliff.
    """
    from app.agent.trading.application.nodes import ANALYST_OUTPUTS

    coverage = sum(1 for key in ANALYST_OUTPUTS.values() if state.get(key)) / len(ANALYST_OUTPUTS)
    mean_spread = sum(e.normalized_spread for e in ledger) / len(ledger) if ledger else 0.0
    risk_turns: list[RiskTurn] = state.get("risk_turns") or []
    flags = sum(len(t.guard_flags) for t in debate_turns) + sum(
        len(t.guard_flags) for t in risk_turns
    )
    score = 0.6 * coverage + 0.3 * (1 - mean_spread) + 0.1 * max(0.0, 1 - flags / 10)
    return round(max(0.0, min(1.0, score)), 2)


# ---------------------------------------------------------------------------
# Numeric guard
# ---------------------------------------------------------------------------

def _grounded_corpus(state) -> str:
    """The analyst reports, and NOTHING the debate or the risk panel wrote.

    This is the only text in a run that traces to something outside the
    models' own reasoning: fundamentals from EDGAR retrieval, technical from
    computed indicators, news from the Finnhub feed. A figure that appears
    here has a source. A figure that appears only downstream of here was
    produced by a model, whatever else is true of it.
    """
    from app.agent.trading.infrastructure.debate_port import report_texts

    return "\n".join(report_texts(state).values())


def _derived_corpus(state, ledger: list[RiskLedgerEntry], debate_turns: list[DebateTurn]) -> str:
    """Everything the debate and the risk panel produced.

    Kept SEPARATE from `_grounded_corpus` rather than concatenated with it,
    because merging the two is what let a fabricated figure certify itself —
    see `verify_decision_memo`.
    """
    claims = canonical_claims(debate_turns)
    parts = [f"{c.text} {c.evidence_quote}" for c in claims.values()]
    parts += [f"{e.text} {e.trigger} {e.evidence_quote}" for e in ledger]
    risk_turns: list[RiskTurn] = state.get("risk_turns") or []
    parts += [s.rationale for t in risk_turns for s in t.payload.scores]
    return "\n".join(parts)


def _numeric_corpus(state, ledger: list[RiskLedgerEntry], debate_turns: list[DebateTurn]) -> str:
    """Grounded + derived, the corpus each GENERATION-time guard checks
    against. Generation-time is the right place for the union: a debater
    citing the previous turn's figure is doing its job, and so is a risk
    persona scoring a factor another persona proposed. The union is only
    wrong for the post-hoc memo check, which is the one place a number has
    finished travelling and its ORIGIN is the question."""
    return _grounded_corpus(state) + "\n" + _derived_corpus(state, ledger, debate_turns)


def _numeric_guard(block_text: str, other_text: str, corpus: str) -> tuple[list[str], list[str]]:
    """Returns (block_flags, gap_flags). Exact containment, not
    tolerance-band matching — see debate_port's module docstring for why
    tolerance bands go blind on dense numeric text."""
    from app.agent.trading.infrastructure.debate_port import _flag_debate_numbers

    return (
        _flag_debate_numbers(block_text, corpus),
        _flag_debate_numbers(other_text, corpus),
    )


# ---------------------------------------------------------------------------
# Post-hoc memo verification — Phase 7. `_numeric_guard` above checks each
# call's own payload fields DURING generation; nothing previously re-checked
# the FULLY ASSEMBLED memo synthesizer_node actually returns (majority-of-N
# voting picks one sample's narrative and Python appends data_gaps/
# verdict_samples on top of it — nothing re-verified that result). This is
# the same containment methodology as `_numeric_guard`, not a second
# implementation, run once more over the assembled whole rather than one
# call's fragment — deliberately not app/application/citation_verifier's
# tolerance-band matching (see debate_port.py's module docstring for why
# that goes blind on a corpus this dense: bands overlap and a fabricated
# figure lands inside somebody's band).
# ---------------------------------------------------------------------------

@dataclass
class MemoVerification:
    passed: bool
    unbacked_numbers: list[str] = field(default_factory=list)
    unresolved_references: list[str] = field(default_factory=list)
    # Figures with NO source: absent from the analyst reports, present only
    # in text the debate or the risk panel wrote. Reported separately from
    # `unbacked_numbers` (absent from everywhere) because the two need
    # different reading. An unbacked number is a memo-assembly bug. A
    # debate-originated one is a number the pipeline produced and then
    # treated as evidence — which may be a sound derivation from grounded
    # endpoints, or may be an invention. Only a human can tell those apart,
    # so this reports rather than blocks. See `verify_decision_memo`.
    debate_originated_numbers: list[str] = field(default_factory=list)


def verify_decision_memo(
    memo: DecisionMemo,
    state,
    ledger: list[RiskLedgerEntry],
    debate_turns: list[DebateTurn],
) -> MemoVerification:
    """`state`/`ledger`/`debate_turns` must be the SAME trial the memo was
    generated from — under majority-of-N sampling each trial ran its own
    independent risk panel with its own factor ids, so verifying against a
    different trial's ledger would misreport a real citation as unresolved.

    `data_gaps`/`assumptions`/`evidence` are excluded from the scanned text
    on purpose: `data_gaps` entries quote already-flagged unbacked numbers
    BY DESIGN (that's why they're gaps, not claims), and `evidence` is
    Python-rendered from resolved references, not model prose — scanning
    either would just re-report what the pipeline already knows.

    The NUMERIC scan is deliberately narrower than the REFERENCE scan, for
    the same reason: `run_research_manager`/`run_risk_judge` already split
    their own fields into blocking (`thesis`; `risk_narrative`+`reasoning`)
    and gap-only (`bull_case`/`bear_case`; `watch_items`) — an unbacked
    number in the gap-only fields was ALREADY let through on purpose and
    recorded honestly in `data_gaps`. Live-caught (ASML, 2026-08-26): the
    first version of this function scanned `watch_items` for numbers too,
    uniformly with the blocking fields, and raised `MemoVerificationError`
    on a `watch_items` figure the pipeline had already, correctly, decided
    was a non-fatal gap — re-deriving a STRICTER standard than generation
    time itself uses is exactly the kind of assembly-step inconsistency
    this function exists to catch, and this was an instance of it, not a
    fabrication. Reference resolution has no such split — `resolve_refs`
    runs over every field at generation time regardless of block/gap
    status, so the post-hoc check mirrors that uniformly instead."""
    grounded = _grounded_corpus(state)
    derived = _derived_corpus(state, ledger, debate_turns)
    load_bearing = "\n".join([memo.research_thesis, memo.risk_debate_summary, memo.reasoning])
    all_narrative = "\n".join([
        memo.bull_case, memo.bear_case, memo.research_thesis,
        memo.risk_debate_summary, memo.reasoning, *memo.watch_items,
    ])

    # Two passes over the SAME text against DIFFERENT corpora, and the
    # difference between them is the finding.
    #
    # Against grounded+derived: what the old single-corpus check reported —
    # figures with no antecedent anywhere in the run.
    # Against grounded alone: figures with no antecedent in any ANALYST
    # REPORT. Subtract the first from the second and what remains is the
    # set of numbers the pipeline invented somewhere between the analysts
    # and the memo.
    numeric_flags, _ = _numeric_guard(load_bearing, "", grounded + "\n" + derived)
    ungrounded, _ = _numeric_guard(load_bearing, "", grounded)
    originated = [n for n in ungrounded if n not in set(numeric_flags)]

    claims = canonical_claims(debate_turns)
    ledger_by_id = {e.factor_id: e for e in ledger}
    unresolved = resolve_refs([all_narrative], claims, ledger_by_id)

    return MemoVerification(
        # `originated` deliberately does NOT gate — see the field comment on
        # MemoVerification and the docstring above. Blocking on it would fail
        # every memo that states a growth rate.
        passed=not numeric_flags and not unresolved,
        unbacked_numbers=numeric_flags,
        unresolved_references=unresolved,
        debate_originated_numbers=originated,
    )


# ---------------------------------------------------------------------------
# The call — shared machinery for both roles
# ---------------------------------------------------------------------------

def _submit_tool(name: str, description: str, payload_cls) -> dict:
    from app.agent.trading.infrastructure.debate_port import _inline_refs

    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": _inline_refs(payload_cls.model_json_schema()),
    }


# Built lazily (a function, not a module-level constant) rather than eagerly
# at import time — same reason every debate_port import in this module is
# local to a function: computing this eagerly here calls _submit_tool ->
# debate_port -> application.nodes at MODULE LOAD time, and nodes.py is
# what imports run_synthesis FROM this module, so an eager build here
# completes the cycle before either module finishes loading.
def _research_manager_tool() -> dict:
    return _submit_tool(
        "submit_research_synthesis", "Submit the debate synthesis. Call exactly once.",
        ResearchManagerPayload,
    )


def _risk_judge_tool() -> dict:
    return _submit_tool(
        "submit_risk_judgment", "Submit the final risk judgment. Call exactly once.",
        RiskJudgePayload,
    )


async def _call_model(
    client: LLMClient, model: str, tool: dict, system_blocks: list[dict],
    messages: list[dict], temperature: float | None,
):
    from app.agent.trading.infrastructure.debate_port import (
        create_with_temperature_fallback,
        reasoning_config,
    )

    return await create_with_temperature_fallback(
        client,
        model=model,
        max_tokens=SYNTHESIS_MAX_TOKENS,
        **reasoning_config(model, temperature),
        system=system_blocks,
        messages=messages,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"], "disable_parallel_tool_use": True},
    )


def _tool_block(response):
    return next((b for b in response.content if b.type == "tool_use"), None)


def _extract(response, payload_cls, tool_name: str):
    block = _tool_block(response)
    if block is None:
        raise ValidationError.from_exception_data(
            payload_cls.__name__, [{"type": "missing", "loc": (tool_name,), "input": None}]
        )
    return payload_cls.model_validate(block.input)


_CORRECTION = (
    "That submission did not validate:\n{error}\n\n"
    "Call {tool} once more, correcting exactly those fields. Change nothing else."
)


def _retry_messages(messages: list[dict], response, error: Exception, tool_name: str) -> list[dict]:
    turns = list(messages)
    if response.content:
        turns.append({"role": "assistant", "content": response.content})
    correction = _CORRECTION.format(error=error, tool=tool_name)
    results = [
        {"type": "tool_result", "tool_use_id": b.id, "is_error": True, "content": correction}
        for b in response.content
        if b.type == "tool_use"
    ]
    turns.append({"role": "user", "content": results or correction})
    return turns


def _accumulate(usage: UsageSummary, raw) -> None:
    usage.input_tokens += raw.input_tokens
    usage.cache_write_tokens += raw.cache_creation_input_tokens or 0
    usage.cache_read_tokens += raw.cache_read_input_tokens or 0
    usage.output_tokens += raw.output_tokens


async def _call_with_schema_retry(
    client: LLMClient, model: str, tool: dict, payload_cls,
    system_blocks: list[dict], messages: list[dict], usage: UsageSummary,
    temperature: float | None,
):
    response = await _call_model(client, model, tool, system_blocks, messages, temperature)
    _accumulate(usage, response.usage)
    try:
        return _extract(response, payload_cls, tool["name"])
    except ValidationError as first:
        block = _tool_block(response)
        print(
            f"[synthesis] {tool['name']}: schema violation, one retry — "
            f"stop_reason={response.stop_reason} "
            f"keys={sorted(block.input) if block else None} "
            f"— {'; '.join(str(first).splitlines()[1:5])}"
        )
        retry_messages = _retry_messages(messages, response, first, tool["name"])
        retry = await _call_model(client, model, tool, system_blocks, retry_messages, temperature)
        _accumulate(usage, retry.usage)
        return _extract(retry, payload_cls, tool["name"])   # a second failure raises out


async def _resolve_with_retry(
    client, model, tool, payload_cls, system_blocks, messages, usage, temperature,
    texts_fn, claims, ledger_by_id, label: str,
):
    """Runs the schema-retry call, then checks references; on an
    unresolved id, retries ONCE more with the bad id named back to the
    model, then raises. Shared between Research Manager and Risk Judge."""
    payload = await _call_with_schema_retry(
        client, model, tool, payload_cls, system_blocks, messages, usage, temperature
    )
    unresolved = resolve_refs(texts_fn(payload), claims, ledger_by_id)
    if unresolved:
        print(f"[synthesis] {label}: unresolved reference(s), one retry: {unresolved}")
        correction = (
            f"These citations do not match any real claim_id or risk factor id "
            f"in the pack you were given: {unresolved}. Call {tool['name']} again, "
            f"citing only ids that actually appear in the pack above. Change nothing else."
        )
        retry_messages = messages + [
            {"role": "assistant", "content": [{"type": "text", "text": "(prior attempt)"}]},
            {"role": "user", "content": correction},
        ]
        payload = await _call_with_schema_retry(
            client, model, tool, payload_cls, system_blocks, retry_messages, usage, temperature
        )
        unresolved = resolve_refs(texts_fn(payload), claims, ledger_by_id)
        if unresolved:
            raise SynthesisReferenceError(
                f"{label} still cites unresolved reference(s) after one correction "
                f"attempt: {unresolved}"
            )
    return payload


def _assert_within_budget(ticker: str, total_cost: float) -> None:
    if total_cost > SYNTHESIS_BUDGET_USD:
        raise AssertionError(
            f"synthesis cost ${total_cost:.4f} for {ticker} exceeds the "
            f"${SYNTHESIS_BUDGET_USD:.2f} combined Research Manager + Risk Judge "
            f"budget — check the model routing and evidence pack size before rerunning"
        )


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------

async def run_research_manager(
    state,
    *,
    claims: dict[str, DebateClaim],
    client: LLMClient | None = None,
    temperature: float | None = None,
) -> tuple[ResearchManagerPayload, float | None, list[str], CostEvent]:
    """Synthesizes the bull/bear debate ONLY. Returns (payload, cost,
    gap_flags, cost_event) — `gap_flags` are unbacked numbers OUTSIDE
    `thesis` (bull_case/bear_case), non-fatal, for the caller to fold into
    the memo's data_gaps.

    `temperature`: None in production (adaptive thinking stays on). Set
    explicitly only by the determinism/stability check scripts — see
    debate_port.reasoning_config for why an explicit temperature disables
    thinking rather than combining with it.
    """
    _maybe_crash("research")
    ticker = state["ticker"]
    client = client or get_client(RESEARCH_MANAGER_MODEL)

    pack = build_research_pack(state)
    system_blocks = [
        {"type": "text", "text": RESEARCH_MANAGER_SYSTEM},
        {"type": "text", "text": pack, "cache_control": {"type": "ephemeral"}},
    ]
    messages: list[dict] = [{"role": "user", "content": "Produce the debate synthesis now."}]
    usage = UsageSummary()

    payload = await _resolve_with_retry(
        client, RESEARCH_MANAGER_MODEL, _research_manager_tool(), ResearchManagerPayload,
        system_blocks, messages, usage, temperature,
        texts_fn=lambda p: [p.bull_case, p.bear_case, p.thesis],
        claims=claims, ledger_by_id={}, label="research_manager",
    )

    # log_cost runs BEFORE the fabrication guard below, not after: usage is
    # already final at this point (no more API calls happen past this line),
    # and a call that trips the guard still spent real tokens. Logging after
    # the guard's raise meant a blocked call's spend never reached
    # cost-log.jsonl — see trading-agent-known-gaps.md (FIG, 2026-08-26).
    event_id = new_event_id("research_manager")
    cost = log_cost(
        ticker, "trading-research-manager", usage,
        model=RESEARCH_MANAGER_MODEL, run_id=state.get("run_id"), event_id=event_id,
    )
    cost_event = record_cost_event(event_id, "research_manager", usage, RESEARCH_MANAGER_MODEL, cost)

    debate_turns: list[DebateTurn] = state.get("debate_turns") or []
    corpus = _numeric_corpus(state, [], debate_turns)
    block_flags, gap_flags = _numeric_guard(
        payload.thesis, payload.bull_case + "\n" + payload.bear_case, corpus
    )
    if block_flags:
        raise SynthesisFabricationError(
            f"Research Manager synthesis for {ticker} has unbacked number(s) in "
            f"`thesis`: {block_flags} — not present in any report or debate claim",
            cost_events=[cost_event],
        )
    if gap_flags:
        # Surfaced by the caller (run_synthesis) into the memo's data_gaps —
        # this function only reports it, it does not have the gaps list to
        # append to.
        print(f"[synthesis] research_manager: unbacked number(s) outside thesis: {gap_flags}")

    return payload, cost, gap_flags, cost_event


# ---------------------------------------------------------------------------
# Risk Judge
# ---------------------------------------------------------------------------

async def run_risk_judge(
    state,
    *,
    ledger: list[RiskLedgerEntry],
    claims: dict[str, DebateClaim],
    research: ResearchManagerPayload,
    client: LLMClient | None = None,
    temperature: float | None = None,
) -> tuple[RiskJudgePayload, float | None, list[str], CostEvent]:
    """Synthesizes the risk ledger, reviews the Research Manager's output,
    and issues the FINAL verdict. Returns (payload, cost, gap_flags, cost_event).

    `temperature`: same contract as run_research_manager — None in
    production, explicit only for the determinism/stability checks.
    """
    _maybe_crash("risk_judge")
    ticker = state["ticker"]
    client = client or get_client(RISK_JUDGE_MODEL)
    ledger_by_id = {e.factor_id: e for e in ledger}

    pack = build_risk_judge_pack(state, ledger, research)
    system_blocks = [
        {"type": "text", "text": RISK_JUDGE_SYSTEM},
        {"type": "text", "text": pack, "cache_control": {"type": "ephemeral"}},
    ]
    messages: list[dict] = [{"role": "user", "content": "Produce the final risk judgment now."}]
    usage = UsageSummary()

    payload = await _resolve_with_retry(
        client, RISK_JUDGE_MODEL, _risk_judge_tool(), RiskJudgePayload,
        system_blocks, messages, usage, temperature,
        texts_fn=lambda p: [p.risk_narrative, p.reasoning, *p.watch_items],
        claims=claims, ledger_by_id=ledger_by_id, label="risk_judge",
    )

    # See the identical comment in run_research_manager: log the cost before
    # the fabrication guard can raise, since usage is already final and a
    # blocked call still spent real tokens.
    event_id = new_event_id("risk_judge")
    cost = log_cost(
        ticker, "trading-risk-judge", usage,
        model=RISK_JUDGE_MODEL, run_id=state.get("run_id"), event_id=event_id,
    )
    cost_event = record_cost_event(event_id, "risk_judge", usage, RISK_JUDGE_MODEL, cost)

    debate_turns: list[DebateTurn] = state.get("debate_turns") or []
    corpus = _numeric_corpus(state, ledger, debate_turns)
    block_flags, gap_flags = _numeric_guard(
        payload.risk_narrative + "\n" + payload.reasoning,
        "\n".join(payload.watch_items),
        corpus,
    )
    if block_flags:
        raise SynthesisFabricationError(
            f"Risk Judge synthesis for {ticker} has unbacked number(s) in "
            f"risk_narrative/reasoning: {block_flags} — not present in any report, "
            f"debate claim, or risk factor",
            cost_events=[cost_event],
        )

    return payload, cost, gap_flags, cost_event


# ---------------------------------------------------------------------------
# Orchestration — both calls, in sequence, assembled into one DecisionMemo
# ---------------------------------------------------------------------------

async def run_synthesis(
    state,
    *,
    ledger: list[RiskLedgerEntry],
    base_gaps: list[str],
    base_evidence: list[str],
    as_of,
    client: LLMClient | None = None,
    research_temperature: float | None = None,
    risk_temperature: float | None = None,
) -> DecisionMemo:
    """Runs the Research Manager, then the Risk Judge (which reviews the
    Research Manager's output), then assembles the final DecisionMemo.

    `research_temperature`/`risk_temperature` are None in every production
    call (nodes.py never passes one) — set only by
    scripts/risk_determinism_check.py, independently, since the two roles
    use different models and the determinism/stability claim is specifically
    about the Risk Judge's verdict, not the Research Manager's lean.
    """
    ticker = state["ticker"]
    debate_turns: list[DebateTurn] = state.get("debate_turns") or []
    claims = canonical_claims(debate_turns)

    research, research_cost, research_gaps, research_cost_event = await run_research_manager(
        state, claims=claims, client=client, temperature=research_temperature
    )
    try:
        risk_judgment, risk_cost, risk_gaps, risk_cost_event = await run_risk_judge(
            state, ledger=ledger, claims=claims, research=research,
            client=client, temperature=risk_temperature,
        )
    except (SynthesisFabricationError, SynthesisReferenceError) as exc:
        # The Research Manager's call already succeeded and its cost is
        # already known here — merge it in before re-raising, or it is lost
        # the moment this exception unwinds past this function. See both
        # exception classes' docstrings.
        exc.cost_events = [research_cost_event, *exc.cost_events]
        raise

    total_cost = (research_cost or 0.0) + (risk_cost or 0.0)
    _assert_within_budget(ticker, total_cost)

    confidence = compute_confidence(state, ledger, debate_turns)
    technical = state.get("technical_report")
    ledger_by_id = {e.factor_id: e for e in ledger}

    data_gaps = list(base_gaps)
    if research_gaps:
        data_gaps.append(
            f"{len(research_gaps)} number(s) in the Research Manager's bull/bear "
            f"case did not appear in any source and may be fabricated: "
            f"{', '.join(research_gaps[:5])}"
        )
    if risk_gaps:
        data_gaps.append(
            f"{len(risk_gaps)} number(s) in the Risk Judge's watch items did not "
            f"appear in any source and may be fabricated: {', '.join(risk_gaps[:5])}"
        )

    evidence = (
        base_evidence
        + _render_evidence([research.bull_case, research.bear_case, research.thesis], claims, {})
        + _render_evidence(
            [risk_judgment.risk_narrative, risk_judgment.reasoning, *risk_judgment.watch_items],
            claims, ledger_by_id,
        )
    )

    return DecisionMemo(
        ticker=ticker,
        bull_case=research.bull_case,
        bear_case=research.bear_case,
        research_thesis=research.thesis,
        risk_debate_summary=risk_judgment.risk_narrative,
        technical_signal=(
            technical.interpretation
            if technical is not None
            else "NOT RUN — technical analyst was excluded from this run"
        ),
        reasoning=risk_judgment.reasoning,
        watch_items=risk_judgment.watch_items,
        # `risk_judgment.verdict` is an `IndividualVerdict` (buy/sell/hold
        # only, see that type's docstring) — this function always produces
        # ONE sample's memo. `Verdict.UNRESOLVED` is never assigned here;
        # it is computed by `application/nodes.py`'s majority-of-N sampling
        # from several calls to this function, one of which it then reuses
        # or overrides. Same string values, so the conversion is lossless.
        verdict=Verdict(risk_judgment.verdict.value),
        confidence=confidence,
        data_as_of_date=as_of,
        data_gaps=data_gaps,
        assumptions=[],
        evidence=evidence,
        cost_events=[research_cost_event, risk_cost_event],
    )
