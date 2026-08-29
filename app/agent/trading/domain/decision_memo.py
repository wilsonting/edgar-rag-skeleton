from datetime import date
from enum import Enum

from pydantic import BaseModel, Field

from app.agent.trading.domain.budget import CostEvent


class Verdict(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    # Added 2026-08-26 (code review, post-fix AVGO+ASML re-measurement):
    # NEVER emitted by a single Risk Judge call — see `IndividualVerdict`
    # below, which is what the tool schema actually offers the model.
    # `UNRESOLVED` is computed in Python by `application/nodes.py`'s
    # majority-of-N sampling: N independent (panel, Research Manager, Risk
    # Judge) trials are run over the same fixed debate, and the memo's
    # verdict is the majority of their individual verdicts, or UNRESOLVED
    # when there is no majority. This exists because post-fix determinism
    # measurement showed the split is real, not an identity artifact: two
    # tickers (AVGO, ASML), both re-measured after slate-identity and
    # threshold-brittleness fixes, still split verdict direction at
    # temperature=0 and at production temperature. See
    # trading-agent-known-gaps.md.
    UNRESOLVED = "unresolved"


class IndividualVerdict(str, Enum):
    """The three choices ONE Risk Judge call may pick — deliberately
    narrower than `Verdict`. `RiskJudgePayload.verdict` uses this type
    (not `Verdict`) so the tool schema sent to the model never lists
    `unresolved` as an option: no single call decides non-resolution, only
    the aggregate over several does. Same string values as `Verdict`'s
    first three members, so `Verdict(payload.verdict.value)` converts
    losslessly wherever a resolved individual verdict needs to become the
    memo's."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class ResearchManagerPayload(BaseModel):
    """EXACTLY what the Research Manager LLM returns. It synthesizes the
    bull/bear DEBATE ONLY — it never sees the risk-panel ledger, and it
    issues no verdict (see the field-level comment below for why not — the
    Risk Judge is the pipeline's sole decision maker). No numbers, no
    quotes: every narrative field cites `[C:claim_id]`; Python resolves
    those and renders their evidence, never retyping a figure the model
    wrote. See infrastructure/synthesis_port.py.
    """

    bull_case: str = Field(description="<=150 words. Cite claims as [C:claim_id].")
    bear_case: str = Field(description="<=150 words. Cite claims as [C:claim_id].")
    thesis: str = Field(
        description=(
            "<=200 words. The debate-level synthesis: which side's evidence is "
            "stronger and why, on the debate alone. Every load-bearing sentence "
            "carries a citation."
        )
    )
    # Deliberately NO verdict field here. Removed 2026-08-26 (code review):
    # a `preliminary_verdict` was an intermediate discrete emission nothing
    # downstream required — the Risk Judge is the only agent whose verdict
    # is used — and it was shown to the Judge as prior context
    # (`_render_research_output` in synthesis_port.py), which is an
    # anchoring effect on the agent that actually decides. Measured live
    # (ASML, 2026-08-26): the preliminary verdict alone flip-flopped
    # sell/hold/sell across three production-temperature samples of the
    # SAME fixed debate, making it a pure noise source with nothing
    # downstream requiring it to exist. The Research Manager's job is
    # cases, not a verdict — that's the Risk Judge's.


class RiskJudgePayload(BaseModel):
    """EXACTLY what the Risk Judge LLM returns. It reviews the risk-panel
    ledger AND the Research Manager's bull/bear thesis, and its own
    `verdict` is the pipeline's sole and FINAL answer — there is no prior
    verdict to affirm or override (see `ResearchManagerPayload`'s
    docstring). Cites `[RFnn]` for risk factors; may also cite
    `[C:claim_id]` in `reasoning` when drawing on the underlying debate.
    No numbers, no quotes, same rule as the Research Manager.
    """

    risk_narrative: str = Field(description="<=200 words. Cite factors as [RF1A2B].")
    reasoning: str = Field(
        description=(
            "<=250 words. The FINAL reasoning behind `verdict` — weigh the risk "
            "ledger against the Research Manager's thesis and state plainly why "
            "this verdict follows. Every load-bearing sentence cited."
        )
    )
    watch_items: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Observables that would change this read. Each must cite [RF1A2B].",
    )
    verdict: IndividualVerdict = Field(description="THE final verdict. buy/sell/hold — no fourth option.")


class DecisionMemo(BaseModel):
    ticker: str
    bull_case: str
    bear_case: str
    # The Research Manager's own debate-level synthesis, kept distinct from
    # `reasoning` (the Risk Judge's FINAL reasoning) so a reader can see the
    # bull/bear case separately from the risk-adjusted decision, rather
    # than that decision being folded invisibly into one paragraph. See
    # synthesis_port.py's two-call split. No `research_preliminary_verdict`
    # field — removed alongside `ResearchManagerPayload.preliminary_verdict`
    # (see that class's docstring); `verdict` below is the Risk Judge's
    # alone, with nothing upstream of it to have "overridden".
    research_thesis: str
    risk_debate_summary: str
    technical_signal: str
    reasoning: str
    # Renamed from `suggested_strategy` (Phase 6 plan §8.4, option 1). The
    # project's architecture deliberately excludes trade execution and any
    # fund-manager agent; `suggested_strategy` was the field most likely to
    # drift into actionable advice and quietly reintroduce what that
    # exclusion rules out.
    watch_items: list[str] = []
    verdict: Verdict
    # The raw per-sample verdicts `verdict` was computed from — e.g.
    # ["hold", "sell", "hold"] — empty when the risk panel didn't run at
    # all (no ledger to sample) rather than when sampling ran and produced
    # only one entry; a single-element list would be a bug, not a real
    # state, since sampling always runs N>=2 when it runs. Added alongside
    # `Verdict.UNRESOLVED` so a reader sees the actual split behind an
    # UNRESOLVED (or a majority) verdict, not a bare label — see
    # `application/nodes.py`'s majority-of-N sampling.
    verdict_samples: list[str] = []
    confidence: float
    data_as_of_date: date
    data_gaps: list[str] = []
    assumptions: list[str] = []
    evidence: list[str] = []
    # The Research Manager's and Risk Judge's own CostEvents (Phase 8) —
    # NOT the majority-of-N sampling's other trials' cost, which the
    # synthesizer node collects separately and returns as its own state
    # delta, since a dropped/non-winning trial's spend is real even though
    # its memo never reaches this field. See application/nodes.py's
    # synthesizer_node.
    cost_events: list[CostEvent] = []
