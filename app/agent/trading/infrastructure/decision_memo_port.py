"""Write the run's overall summary — the DecisionMemo — to the vault."""

from __future__ import annotations

import json
from pathlib import Path

from app.agent.researcher import _save_output
from app.agent.trading.domain.budget import RunTermination, total_spend
from app.agent.trading.domain.decision_memo import DecisionMemo, Verdict

# Fields the pipeline has not implemented yet still carry the literal
# "STUB" from the synthesizer. Rendering that verbatim under a confident
# heading reads like a finding, so the markdown marks it as not-yet-built
# instead. The raw JSON block keeps the unedited value either way.
_STUB = "STUB"


def _render(value: str) -> str:
    if value.strip().startswith(_STUB):
        detail = value.strip()[len(_STUB):].lstrip(" —-")
        return f"*Not yet implemented{f' ({detail})' if detail else ''}.*"
    return value


def _confidence_band(confidence: float) -> str:
    """Display label only — the raw float (`compute_confidence` in
    synthesis_port.py) is still shown alongside it. Thirds, not calibrated:
    same "declared prior, not a finding" status as the weights that
    produce `confidence` itself."""
    if confidence < 1 / 3:
        return "LOW"
    if confidence < 2 / 3:
        return "MEDIUM"
    return "HIGH"


def _render_verdict_line(memo: DecisionMemo) -> str:
    """`verdict_samples` is empty when no risk panel ran to sample (the
    Risk Judge's single call is genuinely the sole decision then) and
    populated whenever `application/nodes.py`'s majority-of-N sampling
    ran — in which case the verdict is Python's aggregate over several
    Judge calls, not any one of them alone, and the label should say so
    rather than keep crediting a single call. See `Verdict.UNRESOLVED`'s
    docstring for why sampling exists at all."""
    if not memo.verdict_samples:
        return (
            f"**Verdict (Risk Judge, sole decision maker):** "
            f"{memo.verdict.value.upper()}"
        )
    samples = ", ".join(memo.verdict_samples)
    basis = "no majority" if memo.verdict == Verdict.UNRESOLVED else "majority"
    return (
        f"**Verdict:** {memo.verdict.value.upper()} "
        f"({basis} of {len(memo.verdict_samples)} samples: {samples})"
    )


def _format_memo_markdown(memo: DecisionMemo) -> str:
    lines = [
        f"# {memo.ticker} — Decision Memo",
        f"{_render_verdict_line(memo)}  ·  **Confidence:** "
        f"{_confidence_band(memo.confidence)} ({memo.confidence:.2f})",
        f"**Data as of:** {memo.data_as_of_date}",
        "",
    ]

    if memo.confidence == 0.0:
        lines += [
            "> **Treat this memo with extreme caution.** Confidence computed at "
            "0.00 — some combination of zero analyst coverage, full ledger "
            "contestation, and heavy guard-flag density. The verdict below is "
            "the Risk Judge's real output, not a placeholder, but the observables "
            "confidence is built from say this run saw very little to go on.",
            "",
        ]

    lines += [
        "## Reasoning",
        "",
        _render(memo.reasoning),
        "",
        "## Bull case",
        "",
        _render(memo.bull_case),
        "",
        "## Bear case",
        "",
        _render(memo.bear_case),
        "",
        "## Research Manager's thesis",
        "",
        _render(memo.research_thesis),
        "",
        "## Technical signal",
        "",
        _render(memo.technical_signal),
        "",
        "## Risk debate",
        "",
        _render(memo.risk_debate_summary),
        "",
        f"## Watch items ({len(memo.watch_items)})",
        "",
    ]
    lines += [f"- {w}" for w in memo.watch_items] if memo.watch_items else ["_None recorded._"]
    lines += [""]

    # Gaps come before evidence deliberately: what the memo did not see
    # bounds what its evidence can support.
    lines += [f"## Data gaps ({len(memo.data_gaps)})", ""]
    lines += [f"- {g}" for g in memo.data_gaps] if memo.data_gaps else ["_None recorded._"]
    lines += ["", f"## Assumptions ({len(memo.assumptions)})", ""]
    lines += [f"- {a}" for a in memo.assumptions] if memo.assumptions else ["_None recorded._"]
    lines += ["", f"## Evidence ({len(memo.evidence)})", ""]
    lines += [f"- {e}" for e in memo.evidence] if memo.evidence else ["_None recorded._"]

    # The unedited object, so the file is a faithful record of what the
    # pipeline produced and not only of how it was rendered.
    lines += [
        "",
        "## Raw memo",
        "",
        "```json",
        json.dumps(memo.model_dump(mode="json"), indent=2),
        "```",
    ]
    return "\n".join(lines)


def save_decision_memo(
    memo: DecisionMemo, provenance: str | None = None
) -> Path:
    return _save_output(
        _format_memo_markdown(memo),
        memo.ticker.upper(),
        "decision",
        provenance=provenance,
    )


def _format_failed_memo_markdown(memo: DecisionMemo, verification) -> str:
    """`verification` is a `MemoVerification` (synthesis_port.py) — kept as
    a loose type here rather than imported, to avoid decision_memo_port
    (infrastructure) depending on synthesis_port (also infrastructure) for
    what is really just two lists of strings."""
    banner = [
        "> [!FAILED] VERIFICATION FAILED — DO NOT ACT ON THIS MEMO",
        ">",
        "> The memo below failed Phase 7's post-hoc check. Every per-call "
        "guard passed during generation, so this signals an ASSEMBLY-STEP "
        "BUG, not an ordinary model fabrication — treat this as a pipeline "
        "defect to investigate, not a tradeable memo.",
        ">",
    ]
    if verification.unbacked_numbers:
        banner.append(f"> **Unbacked number(s):** {', '.join(verification.unbacked_numbers)}")
    if verification.unresolved_references:
        banner.append(
            f"> **Unresolved reference(s):** {', '.join(verification.unresolved_references)}"
        )
    # Not a cause of the failure — these never gate — but a reader debugging
    # a failed assembly wants to know which figures had no analyst source.
    if getattr(verification, "debate_originated_numbers", None):
        banner.append(
            f"> **No analyst source (debate-originated):** "
            f"{', '.join(verification.debate_originated_numbers)}"
        )
    banner += ["", "---", ""]
    return "\n".join(banner) + _format_memo_markdown(memo)


def save_failed_decision_memo(
    memo: DecisionMemo, verification, provenance: str | None = None
) -> Path:
    """Written from INSIDE synthesizer_node, before it raises
    MemoVerificationError — same pattern technical_node/fundamentals_node
    already use to save their own output while the graph is still
    executing. The memo that failed verification is the debugging
    artifact; discarding it re-runs the pipeline blind."""
    return _save_output(
        _format_failed_memo_markdown(memo, verification),
        memo.ticker.upper(),
        "decision_failed",
        provenance=provenance,
    )


def _format_aborted_run_markdown(state, terminated_by: RunTermination) -> str:
    """No DecisionMemo to render — a budget/deadline abort can fire before
    any analyst report exists, and DecisionMemo's required fields (verdict,
    confidence, ...) assume a completed run. This is deliberately a
    different, looser shape: which stages the run reached before it stopped,
    and why, never a verdict-shaped stand-in for one it never reached."""
    events = state.get("cost_events") or []
    budget = state.get("budget")
    stages = [
        ("fundamentals", state.get("fundamentals_report") is not None),
        ("technical", state.get("technical_report") is not None),
        ("news/sentiment", state.get("news_digest") is not None),
        ("debate", len(state.get("debate_turns") or []) > 0),
        ("risk panel", len(state.get("risk_turns") or []) > 0),
        ("decision memo", state.get("decision_memo") is not None),
    ]
    lines = [
        f"> [!ABORTED] RUN ABORTED — {terminated_by.value.upper()}",
        ">",
        "> This run stopped before producing a verdict. No memo was "
        "synthesized — do not treat the absence of one section below as a "
        "finding about that stage, since a later stage never ran to "
        "produce it.",
        "",
        f"# {state.get('ticker', '?')} — Aborted Run",
        "",
        f"**Terminated by:** {terminated_by.value}",
        f"**Spend at abort:** ${total_spend(events):.4f}"
        + (f" (budget ${budget.max_usd:.2f})" if budget else ""),
        f"**LLM calls logged:** {len(events)}",
        "",
        "## Stages reached",
        "",
    ]
    for name, reached in stages:
        lines.append(f"- [{'x' if reached else ' '}] {name}")
    return "\n".join(lines) + "\n"


def save_aborted_run_memo(state, terminated_by: RunTermination) -> Path:
    """Written from graceful_abort_node — the run-level counterpart to
    save_failed_decision_memo above. Same reason: a run that stopped without
    a verdict is a debugging/audit artifact, not nothing."""
    ticker = state.get("ticker") or "UNKNOWN"
    return _save_output(
        _format_aborted_run_markdown(state, terminated_by),
        ticker.upper(),
        "decision_aborted",
    )
