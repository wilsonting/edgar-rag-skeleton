"""Write the run's overall summary — the DecisionMemo — to the vault."""

from __future__ import annotations

import json
from pathlib import Path

from app.agent.researcher import _save_output
from app.agent.trading.domain.decision_memo import DecisionMemo

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


def _format_memo_markdown(memo: DecisionMemo) -> str:
    lines = [
        f"# {memo.ticker} — Decision Memo",
        f"**Verdict:** {memo.verdict.value.upper()}  ·  "
        f"**Confidence:** {memo.confidence:.2f}",
        f"**Data as of:** {memo.data_as_of_date}",
        "",
    ]

    if memo.confidence == 0.0:
        lines += [
            "> **This memo is not a recommendation.** Confidence is 0.00 and the "
            "synthesis logic is still a stub — the verdict below is a placeholder "
            "the schema requires, not a conclusion drawn from the evidence.",
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
