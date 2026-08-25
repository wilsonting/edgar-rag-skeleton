from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class SynthesisPayload(BaseModel):
    """EXACTLY what the synthesizer LLM returns. No numbers, no quotes, no
    confidence score — those are fabrication surfaces in a node whose whole
    output is what a human reads, so Python fills them in from data the
    model never gets to retype. See infrastructure/synthesis_port.py.

    Every narrative field cites its support with reference tokens —
    `[C:claim_id]` for a debate claim, `[RF00]` for a risk-ledger factor —
    rather than asserting a figure or a quote directly. `resolve_refs`
    checks every token against the real claim/factor ids and raises on
    anything unresolved; `evidence` in the final DecisionMemo is rendered
    by Python from what those tokens resolve to, never retyped by the model.
    """

    bull_case: str = Field(description="<=150 words. Cite claims as [C:claim_id].")
    bear_case: str = Field(description="<=150 words. Cite claims as [C:claim_id].")
    risk_narrative: str = Field(description="<=200 words. Cite factors as [RF00].")
    reasoning: str = Field(
        description="<=250 words. Every load-bearing sentence carries a citation."
    )
    watch_items: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Observables that would change this read. Each must cite [RF00].",
    )
    verdict: Verdict


class DecisionMemo(BaseModel):
    ticker: str
    bull_case: str
    bear_case: str
    risk_debate_summary: str
    technical_signal: str
    reasoning: str
    # Renamed from `suggested_strategy` (Phase 6 plan §8.4, option 1). The
    # project's architecture deliberately excludes trade execution and any
    # fund-manager agent; `suggested_strategy` was the field most likely to
    # drift into actionable advice and quietly reintroduce what that
    # exclusion rules out. `watch_items` is what the field actually held in
    # spirit — observables that would change the read — under a name that
    # can't be misread as a recommendation.
    watch_items: list[str] = []
    verdict: Verdict
    confidence: float
    data_as_of_date: date
    data_gaps: list[str] = []
    assumptions: list[str] = []
    evidence: list[str] = []
