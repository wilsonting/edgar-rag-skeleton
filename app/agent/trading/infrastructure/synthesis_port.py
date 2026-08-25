"""The synthesizer's LLM call: evidence pack, citation resolution, computed
confidence, and the numeric fabrication guard on the rendered memo.

Highest-fabrication-risk node in the pipeline — the only one that can invent
a figure appearing in no upstream report, and the only one a human actually
reads (Phase 6 plan §8). Structured accordingly: `SynthesisPayload` carries
no numbers and no quotes at all. Every narrative sentence cites a reference
token — `[C:claim_id]` for a debate claim, `[RF00]` for a risk-ledger factor
— and Python resolves those, renders the evidence list from what they point
to, and computes confidence from observables. Nothing the model types
directly becomes a number or a quote in the final memo.

Imports from debate_port are LOCAL to each function rather than at module
level, deliberately: debate_port imports `ANALYST_OUTPUTS` from
application.nodes at ITS module level, and nodes.py imports `run_synthesis`
from this module — a top-level import here would complete the cycle
(nodes -> synthesis_port -> debate_port -> nodes, mid-init) and fail before
either module finishes loading. risk_port.py hits the identical shape for
the identical reason and resolves it the same way.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

from anthropic import AsyncAnthropic
from pydantic import ValidationError

from app.agent.researcher import (
    AGENT_MODEL,
    _MODEL_PRICING,
    UsageSummary,
    log_cost,
)
from app.agent.trading.domain.debate import DebateClaim, DebateTurn, canonical_claims
from app.agent.trading.domain.decision_memo import DecisionMemo, SynthesisPayload, Verdict
from app.agent.trading.domain.risk import RiskLedgerEntry, RiskTurn

SYNTHESIS_MODEL = os.getenv("TRADING_SYNTHESIS_MODEL") or AGENT_MODEL
SYNTHESIS_MAX_TOKENS = 5000

# Phase 6 plan §10: one call, input ~17k tokens (pack + 6 debate turns + 6
# risk turns), output ~3k tokens, estimated ≈$0.096 on a $3/$15 basis —
# EXTRAPOLATED, not measured against a live run as of this write-up. Budget
# set with margin over that estimate; re-measure and tighten once a live run
# exists, the same way DEBATE_BUDGET_USD's docstring records what was
# actually measured rather than only projected.
SYNTHESIS_BUDGET_USD = 0.20

SYNTHESIS_THINKING: dict[str, Any] = {"type": "adaptive"}
SYNTHESIS_EFFORT = "low"

if SYNTHESIS_MODEL not in _MODEL_PRICING:
    print(
        f"[synthesis] WARNING: no pricing configured for {SYNTHESIS_MODEL} — "
        f"per-call cost will log as null and the ${SYNTHESIS_BUDGET_USD:.2f} "
        f"budget assertion cannot fire. Add it to _MODEL_PRICING in researcher.py."
    )

_CRASH_AT = os.getenv("SYNTHESIS_CRASH_AT")   # "before" | "after" | None


def _maybe_crash(when: str) -> None:
    if _CRASH_AT != when:
        return
    print(f"[synthesis] FORCED CRASH {when} (SYNTHESIS_CRASH_AT)")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


class SynthesisReferenceError(Exception):
    """A reference token in the synthesis payload names no real claim_id or
    factor_id, on the SECOND attempt (the model was already shown the
    unresolved ids once and asked to correct them). Raised rather than
    dropped: a silently-dropped reference leaves a fluent sentence whose
    support has vanished, which is the exact failure the reference scheme
    exists to prevent — the memo is worth less than the error."""


class SynthesisFabricationError(Exception):
    """A number in `reasoning` or `risk_narrative` — the two load-bearing
    prose fields — appears in no report, no debate claim, and no risk
    factor. Blocking rather than flagging: numbers elsewhere (bull/bear
    case, watch items) are gaps, not blocks — see `_numeric_guard`."""


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = """\
You are the final synthesizer for an equity research pipeline. You will be
given the analyst reports, the bull/bear debate transcript, and the risk
panel's ledger for one ticker. Produce a balanced synthesis.

HARD RULES — checked in code after you answer:

1. You produce LANGUAGE AND JUDGEMENT ONLY. Never write a number, a dollar
   figure, a percentage, or a direct quote in `bull_case`, `bear_case`,
   `risk_narrative`, `reasoning`, or `watch_items`. Cite instead: every claim
   from the debate is `[C:claim_id]`, every risk factor from the ledger is
   `[RFnn]` (the exact id shown in the ledger, e.g. [RF00]). Python resolves
   these into the actual evidence — you never retype a figure that was
   already established elsewhere in the pipeline.
2. Every load-bearing sentence in `reasoning` carries at least one citation.
   An uncited assertion is exactly as unverifiable as a fabricated number.
3. `bull_case` and `bear_case` should draw primarily on the debate's claims;
   `risk_narrative` should draw primarily on the risk ledger's factors.
   Citing a claim_id or factor_id that is not actually in the pack you were
   given will be rejected and cost you a retry — cite only what you see.
4. `watch_items`: at most 5, each an observable (not a vague sentiment) that
   would change this read, each citing the `[RFnn]` factor it comes from.
5. `verdict` is buy/sell/hold. Do not hedge into a fourth option — pick the
   one the evidence on balance supports, and let `reasoning` carry the
   nuance.

A report, debate, or risk panel marked "NOT RUN" / "no risk panel ran" /
"empty" is missing evidence, not neutral evidence — do not infer anything
from its absence.

Call `submit_synthesis` exactly once. Say nothing else."""


# ---------------------------------------------------------------------------
# Evidence pack
# ---------------------------------------------------------------------------

def _render_ledger(ledger: list[RiskLedgerEntry]) -> str:
    if not ledger:
        return "RISK LEDGER: none — no risk panel ran."
    lines = ["RISK LEDGER (cite factor ids exactly as shown, e.g. [RF00]):"]
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


def build_synthesis_pack(state, ledger: list[RiskLedgerEntry]) -> str:
    from app.agent.trading.infrastructure.debate_port import (
        build_evidence_pack,
        render_transcript,
    )

    debate_turns: list[DebateTurn] = state.get("debate_turns") or []
    return (
        build_evidence_pack(state)
        + "\n\nBULL/BEAR DEBATE (cite claims exactly as shown, e.g. [C:claim-id]):\n"
        + render_transcript(debate_turns)
        + "\n\n"
        + _render_ledger(ledger)
    )


# ---------------------------------------------------------------------------
# Reference resolution — §8.2
# ---------------------------------------------------------------------------

_REF_PATTERN = re.compile(r"\[C:([\w.-]+)\]|\[(RF\d+)\]")


def extract_refs(payload: SynthesisPayload) -> list[str]:
    """Every `[C:id]`/`[RFnn]` token across the narrative fields, IN ORDER,
    duplicates included — order and duplication both matter to the caller
    (resolve_refs needs every occurrence checked; _render_evidence needs
    first-occurrence order for a stable citation list)."""
    text = "\n".join(
        [payload.bull_case, payload.bear_case, payload.risk_narrative, payload.reasoning]
        + payload.watch_items
    )
    return [m.group(1) or m.group(2) for m in _REF_PATTERN.finditer(text)]


def resolve_refs(
    payload: SynthesisPayload,
    claims: dict[str, DebateClaim],
    ledger_by_id: dict[str, RiskLedgerEntry],
) -> list[str]:
    """Reference tokens that resolve to neither a real claim_id nor a real
    factor_id. Empty means every citation in the payload is real."""
    known = set(claims) | set(ledger_by_id)
    return sorted({r for r in extract_refs(payload) if r not in known})


def _render_evidence(
    payload: SynthesisPayload,
    claims: dict[str, DebateClaim],
    ledger_by_id: dict[str, RiskLedgerEntry],
) -> list[str]:
    """The memo's citation-derived evidence lines, one per DISTINCT reference
    actually used, first-occurrence order. Rendered entirely from stored
    data — the model's own text never becomes an evidence line."""
    seen: set[str] = set()
    lines: list[str] = []
    for ref in extract_refs(payload):
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
# Confidence — §8.3, computed from observables, never model-emitted
# ---------------------------------------------------------------------------

def compute_confidence(
    state, ledger: list[RiskLedgerEntry], debate_turns: list[DebateTurn]
) -> float:
    """A judgement call in its weights, not a calibration — say so here
    rather than implying precision the number doesn't have. What matters is
    that it is DERIVED from something a reader can point at (analyst
    coverage, ledger contestation, guard-flag density), not a number the
    model typed. Calibrating the weights against realised outcomes is a
    Phase 9 item (Phase 6 plan §11), not attempted here."""
    from app.agent.trading.application.nodes import ANALYST_OUTPUTS

    coverage = sum(1 for key in ANALYST_OUTPUTS.values() if state.get(key)) / len(ANALYST_OUTPUTS)
    contested_share = sum(1 for e in ledger if e.contested) / len(ledger) if ledger else 0.0
    risk_turns: list[RiskTurn] = state.get("risk_turns") or []
    flags = sum(len(t.guard_flags) for t in debate_turns) + sum(
        len(t.guard_flags) for t in risk_turns
    )
    score = 0.6 * coverage + 0.3 * (1 - contested_share) + 0.1 * max(0.0, 1 - flags / 10)
    return round(max(0.0, min(1.0, score)), 2)


# ---------------------------------------------------------------------------
# Numeric guard — §8.5
# ---------------------------------------------------------------------------

def _numeric_corpus(state, ledger: list[RiskLedgerEntry], debate_turns: list[DebateTurn]) -> str:
    from app.agent.trading.infrastructure.debate_port import report_texts

    claims = canonical_claims(debate_turns)
    parts = list(report_texts(state).values())
    parts += [f"{c.text} {c.evidence_quote}" for c in claims.values()]
    parts += [f"{e.text} {e.trigger} {e.evidence_quote}" for e in ledger]
    risk_turns: list[RiskTurn] = state.get("risk_turns") or []
    parts += [s.rationale for t in risk_turns for s in t.payload.scores]
    return "\n".join(parts)


def _numeric_guard(payload: SynthesisPayload, corpus: str) -> tuple[list[str], list[str]]:
    """Returns (block_flags, gap_flags). Two-tier per §8.5: a number in
    `reasoning` or `risk_narrative` is load-bearing and blocks; elsewhere it
    is a data_gaps entry. Exact containment, not tolerance-band matching —
    Phase 5 established why on dense numeric text (debate_port module
    docstring): overlapping bands eventually cover most of the number line
    and the guard goes silent while reading as clean."""
    from app.agent.trading.infrastructure.debate_port import _flag_debate_numbers

    block_text = payload.reasoning + "\n" + payload.risk_narrative
    other_text = payload.bull_case + "\n" + payload.bear_case + "\n" + "\n".join(payload.watch_items)
    return (
        _flag_debate_numbers(block_text, corpus),
        _flag_debate_numbers(other_text, corpus),
    )


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------

def _submit_tool():
    from app.agent.trading.infrastructure.debate_port import _inline_refs

    return {
        "name": "submit_synthesis",
        "description": "Submit the final synthesis. Call exactly once.",
        "strict": True,
        "input_schema": _inline_refs(SynthesisPayload.model_json_schema()),
    }


async def _call_model(client: AsyncAnthropic, system_blocks: list[dict], messages: list[dict]):
    from app.agent.trading.infrastructure.debate_port import supports_adaptive_thinking

    reasoning: dict[str, Any] = {}
    if supports_adaptive_thinking(SYNTHESIS_MODEL):
        reasoning["thinking"] = SYNTHESIS_THINKING
        reasoning["output_config"] = {"effort": SYNTHESIS_EFFORT}

    return await client.messages.create(
        model=SYNTHESIS_MODEL,
        max_tokens=SYNTHESIS_MAX_TOKENS,
        **reasoning,
        system=system_blocks,
        messages=messages,
        tools=[_submit_tool()],
        tool_choice={"type": "tool", "name": "submit_synthesis", "disable_parallel_tool_use": True},
    )


def _tool_block(response):
    return next((b for b in response.content if b.type == "tool_use"), None)


def _extract(response) -> SynthesisPayload:
    block = _tool_block(response)
    if block is None:
        raise ValidationError.from_exception_data(
            "SynthesisPayload", [{"type": "missing", "loc": ("submit_synthesis",), "input": None}]
        )
    return SynthesisPayload.model_validate(block.input)


_CORRECTION = (
    "That submission did not validate:\n{error}\n\n"
    "Call submit_synthesis once more, correcting exactly those fields. Change nothing else."
)


def _retry_messages(messages: list[dict], response, error: Exception) -> list[dict]:
    turns = list(messages)
    if response.content:
        turns.append({"role": "assistant", "content": response.content})
    correction = _CORRECTION.format(error=error)
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
    client: AsyncAnthropic, system_blocks: list[dict], messages: list[dict], usage: UsageSummary
) -> SynthesisPayload:
    """One schema-validation retry, same shape as debate_port/risk_port."""
    response = await _call_model(client, system_blocks, messages)
    _accumulate(usage, response.usage)
    try:
        return _extract(response)
    except ValidationError as first:
        block = _tool_block(response)
        print(
            f"[synthesis] schema violation, one retry — stop_reason="
            f"{response.stop_reason} keys={sorted(block.input) if block else None} "
            f"— {'; '.join(str(first).splitlines()[1:5])}"
        )
        retry_messages = _retry_messages(messages, response, first)
        retry = await _call_model(client, system_blocks, retry_messages)
        _accumulate(usage, retry.usage)
        return _extract(retry)   # a second failure raises out of the node


def _assert_within_budget(ticker: str, cost: float | None) -> None:
    if cost is not None and cost > SYNTHESIS_BUDGET_USD:
        raise AssertionError(
            f"synthesis cost ${cost:.4f} for {ticker} exceeds the "
            f"${SYNTHESIS_BUDGET_USD:.2f} budget — check SYNTHESIS_MODEL routing "
            f"and the evidence pack size before rerunning"
        )


async def run_synthesis(
    state,
    *,
    ledger: list[RiskLedgerEntry],
    base_gaps: list[str],
    base_evidence: list[str],
    as_of,
    client: AsyncAnthropic | None = None,
) -> DecisionMemo:
    """Builds the pack, resolves references (one retry on an unresolved id,
    then fails the run), computes confidence, runs the numeric guard, and
    assembles the final DecisionMemo. `ledger`/`base_gaps`/`base_evidence`
    come from `nodes.py`'s caveat functions — this port owns everything
    downstream of the LLM call, not the news/debate/risk caveat logic
    itself, matching risk_port/debate_port's own split from their nodes.
    """
    _maybe_crash("before")

    ticker = state["ticker"]
    debate_turns: list[DebateTurn] = state.get("debate_turns") or []
    claims = canonical_claims(debate_turns)
    ledger_by_id = {e.factor_id: e for e in ledger}

    pack = build_synthesis_pack(state, ledger)
    client = client or AsyncAnthropic()
    system_blocks = [
        {"type": "text", "text": SYNTHESIS_SYSTEM},
        {"type": "text", "text": pack, "cache_control": {"type": "ephemeral"}},
    ]
    messages: list[dict] = [{"role": "user", "content": "Produce the synthesis now."}]

    usage = UsageSummary()
    payload = await _call_with_schema_retry(client, system_blocks, messages, usage)

    unresolved = resolve_refs(payload, claims, ledger_by_id)
    if unresolved:
        print(f"[synthesis] unresolved reference(s), one retry: {unresolved}")
        correction = (
            f"These citations do not match any real claim_id or risk factor id "
            f"in the pack you were given: {unresolved}. Call submit_synthesis "
            f"again, citing only ids that actually appear in the pack above. "
            f"Change nothing else."
        )
        messages = messages + [
            {"role": "assistant", "content": [{"type": "text", "text": "(prior attempt)"}]},
            {"role": "user", "content": correction},
        ]
        payload = await _call_with_schema_retry(client, system_blocks, messages, usage)
        unresolved = resolve_refs(payload, claims, ledger_by_id)
        if unresolved:
            raise SynthesisReferenceError(
                f"synthesis for {ticker} still cites unresolved reference(s) after "
                f"one correction attempt: {unresolved}"
            )

    corpus = _numeric_corpus(state, ledger, debate_turns)
    block_flags, gap_flags = _numeric_guard(payload, corpus)
    if block_flags:
        raise SynthesisFabricationError(
            f"synthesis for {ticker} has unbacked number(s) in reasoning/"
            f"risk_narrative: {block_flags} — not present in any report, debate "
            f"claim, or risk factor"
        )

    cost = log_cost(ticker, "trading-synthesis", usage, model=SYNTHESIS_MODEL)
    _assert_within_budget(ticker, cost)

    confidence = compute_confidence(state, ledger, debate_turns)
    technical = state.get("technical_report")
    data_gaps = list(base_gaps)
    if gap_flags:
        data_gaps.append(
            f"{len(gap_flags)} number(s) in the bull/bear case or watch items did "
            f"not appear in any source and may be fabricated: {', '.join(gap_flags[:5])}"
        )

    memo = DecisionMemo(
        ticker=ticker,
        bull_case=payload.bull_case,
        bear_case=payload.bear_case,
        risk_debate_summary=payload.risk_narrative,
        technical_signal=(
            technical.interpretation
            if technical is not None
            else "NOT RUN — technical analyst was excluded from this run"
        ),
        reasoning=payload.reasoning,
        watch_items=payload.watch_items,
        verdict=payload.verdict,
        confidence=confidence,
        data_as_of_date=as_of,
        data_gaps=data_gaps,
        assumptions=[],
        evidence=base_evidence + _render_evidence(payload, claims, ledger_by_id),
    )

    _maybe_crash("after")
    return memo
