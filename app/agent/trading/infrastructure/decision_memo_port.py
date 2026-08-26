"""Write the run's overall summary — the DecisionMemo — to the vault."""

from __future__ import annotations

import json
from pathlib import Path

from app.agent.researcher import _save_output
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
        f"{_render_verdict_line(memo)}  ·  **Confidence:** {memo.confidence:.2f}",
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
