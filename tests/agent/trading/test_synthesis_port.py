"""Synthesis port tests — Research Manager + Risk Judge, the two-call split
(Phase 6 gap closure). Phase 6 exit criteria 4 and 5 (reference integrity,
fabrication guard) still apply, now to each call independently, plus the
new override/affirm bookkeeping between them.

Mocked LLM throughout, same fake-client pattern as test_debate_port.py and
test_risk_port.py — no network, no cost.
"""

from __future__ import annotations

from datetime import date

import pytest

import app.agent.trading.infrastructure.synthesis_port as port
from app.agent.trading.domain.debate import DebateClaim, DebateTurn, DebateTurnPayload
from app.agent.trading.domain.decision_memo import ResearchManagerPayload, RiskJudgePayload, Verdict
from app.agent.trading.domain.fundamentals_report import FundamentalsReport
from app.agent.trading.domain.risk import RiskLedgerEntry


class _Block:
    type = "tool_use"

    def __init__(self, payload: dict):
        self.input = payload
        self.id = "toolu_stub"


class _Usage:
    def __init__(self):
        self.input_tokens = 1000
        self.output_tokens = 200
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0


class _Response:
    def __init__(self, payload: dict | None):
        self.content = [] if payload is None else [_Block(payload)]
        self.usage = _Usage()
        self.stop_reason = "tool_use" if payload is not None else "end_turn"


class _FakeMessages:
    def __init__(self, payloads: list[dict | None]):
        self._payloads = list(payloads)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Response(self._payloads.pop(0))


class _FakeClient:
    """One client, one shared payload queue — mirrors how `run_synthesis`
    actually uses a single client across both the Research Manager and the
    Risk Judge calls, so tests can supply payloads in call order."""

    def __init__(self, payloads: list[dict | None]):
        self.messages = _FakeMessages(payloads)


def _research_payload(**overrides) -> dict:
    base = {
        "bull_case": "Margins are stable [C:margin-hold].",
        "bear_case": "Growth is slowing [C:margin-hold].",
        "thesis": "The bull case rests on stable margins [C:margin-hold], on balance a hold.",
    }
    base.update(overrides)
    return base


def _risk_payload(**overrides) -> dict:
    base = {
        "risk_narrative": "Customer concentration is the key risk [RF00].",
        "reasoning": "Affirming the Research Manager's hold [RF00] [C:margin-hold].",
        "watch_items": ["Watch for a break below the trigger level [RF00]."],
        "verdict": "hold",
    }
    base.update(overrides)
    return base


def _debate_turns() -> list[DebateTurn]:
    return [
        DebateTurn(
            turn_index=0, round_num=1, side="bull",
            payload=DebateTurnPayload(
                stance="hold", argument="a",
                claims=[DebateClaim(
                    claim_id="margin-hold", text="Operating margin is stable.",
                    evidence_ref="fundamentals", evidence_quote="operating margin of 34.1%",
                )],
            ),
        )
    ]


def _ledger() -> list[RiskLedgerEntry]:
    return [
        RiskLedgerEntry(
            factor_id="RF00", text="Customer concentration risk.",
            trigger="top customer exceeds 25% of revenue", horizon="quarters",
            evidence_ref="none", evidence_quote="", proposed_by="neutral",
            scores={"aggressive": (4, 3), "conservative": (2, 3)},
            severity_spread=2, likelihood_spread=0, contested=True,
            missing_scores=["neutral"],
        )
    ]


def _state(**extra) -> dict:
    state = {
        "ticker": "ACN",
        "as_of_date": date(2026, 8, 19),
        "fundamentals_report": FundamentalsReport(
            ticker="ACN", summary="Revenue grew to 64.9 billion with an operating margin of 34.1%.",
            input_tokens=0, cache_write_tokens=0, cache_read_tokens=0, output_tokens=0,
            generated_at=date(2026, 8, 19),
        ),
        "debate_turns": _debate_turns(),
        "risk_turns": [],
    }
    state.update(extra)
    return state


@pytest.fixture(autouse=True)
def _no_cost_log(monkeypatch):
    monkeypatch.setattr(port, "log_cost", lambda *a, **k: 0.02)


def _run(payloads, **state_extra):
    """payloads are consumed in call order: research (+ retries), then risk
    judge (+ retries)."""
    client = _FakeClient(payloads)
    state = _state(**state_extra)
    return port.run_synthesis(
        state, ledger=_ledger(), base_gaps=["a base gap"], base_evidence=["a base evidence line"],
        as_of=state["as_of_date"], client=client,
    ), client


# ---------------------------------------------------------------------------
# Pure functions: extract_refs / resolve_refs
# ---------------------------------------------------------------------------

def test_extract_refs_finds_both_reference_forms():
    refs = port.extract_refs("cites [C:margin-hold]", "and [RF00]")
    assert "margin-hold" in refs
    assert "RF00" in refs


def test_resolve_refs_is_empty_when_every_citation_is_real():
    claims = {c.claim_id: c for t in _debate_turns() for c in t.payload.claims}
    ledger_by_id = {e.factor_id: e for e in _ledger()}
    texts = ["cites [C:margin-hold] and [RF00]"]
    assert port.resolve_refs(texts, claims, ledger_by_id) == []


def test_resolve_refs_names_the_unresolved_id():
    claims = {c.claim_id: c for t in _debate_turns() for c in t.payload.claims}
    ledger_by_id = {e.factor_id: e for e in _ledger()}
    texts = ["a latent risk exists [RF99]"]
    assert port.resolve_refs(texts, claims, ledger_by_id) == ["RF99"]


# ---------------------------------------------------------------------------
# Criterion 4 — reference integrity, through run_synthesis end to end.
# Each unresolved-reference case below supplies research payloads that
# validate cleanly (so only the field under test varies) to isolate which
# call's guard fires.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_an_injected_unresolvable_reference_in_the_risk_judge_raises_not_drops():
    bad_risk = _risk_payload(risk_narrative="A latent risk exists [RF99].")
    coro, client = _run([_research_payload(), bad_risk, bad_risk])

    with pytest.raises(port.SynthesisReferenceError, match="RF99"):
        await coro

    # 1 research call + 2 risk judge attempts (initial + one correction retry)
    assert len(client.messages.calls) == 3


@pytest.mark.anyio
async def test_an_injected_unresolvable_reference_in_the_research_manager_raises_not_drops():
    bad_research = _research_payload(thesis="Cites a nonexistent claim [C:does-not-exist].")
    coro, client = _run([bad_research, bad_research])

    with pytest.raises(port.SynthesisReferenceError, match="does-not-exist"):
        await coro

    assert len(client.messages.calls) == 2   # research only — risk judge never runs


@pytest.mark.anyio
async def test_a_corrected_reference_on_retry_succeeds():
    bad_risk = _risk_payload(risk_narrative="A latent risk exists [RF99].")
    coro, _ = _run([_research_payload(), bad_risk, _risk_payload()])

    memo = await coro

    assert memo.verdict == Verdict.HOLD
    assert any("RF00" in e or "Customer concentration" in e for e in memo.evidence)


@pytest.mark.anyio
async def test_the_risk_judge_may_cite_a_debate_claim_too():
    """RiskJudgePayload.reasoning may cite [C:id] as well as [RFnn] — the
    Risk Judge sees the debate, not just the ledger."""
    coro, _ = _run([_research_payload(), _risk_payload(
        reasoning="Affirming based on [C:margin-hold] and [RF00]."
    )])

    memo = await coro

    assert any("margin-hold" in e or "Operating margin" in e for e in memo.evidence)


# ---------------------------------------------------------------------------
# Criterion 5 — fabrication guard, per role
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_an_unbacked_number_in_the_research_thesis_blocks_the_run():
    bait = _research_payload(thesis="Revenue could reach $9,999 million [C:margin-hold].")
    coro, _ = _run([bait])

    with pytest.raises(port.SynthesisFabricationError, match="9999|9,999"):
        await coro


@pytest.mark.anyio
async def test_an_unbacked_number_in_risk_judge_reasoning_blocks_the_run():
    bait = _risk_payload(reasoning="Losses could hit $12,345 thousand [RF00].")
    coro, _ = _run([_research_payload(), bait])

    with pytest.raises(port.SynthesisFabricationError):
        await coro


@pytest.mark.anyio
async def test_an_unbacked_number_in_bull_case_is_a_gap_not_a_block():
    bait = _research_payload(bull_case="Upside could reach $7,777 million [C:margin-hold].")
    coro, _ = _run([bait, _risk_payload()])

    memo = await coro

    assert any("7777" in g or "7,777" in g for g in memo.data_gaps)


@pytest.mark.anyio
async def test_an_unbacked_number_in_watch_items_is_a_gap_not_a_block():
    bait = _risk_payload(watch_items=["A move past $8,888 would change this [RF00]."])
    coro, _ = _run([_research_payload(), bait])

    memo = await coro

    assert any("8888" in g or "8,888" in g for g in memo.data_gaps)


@pytest.mark.anyio
async def test_a_faithfully_cited_number_from_the_pack_is_not_flagged():
    clean_research = _research_payload(thesis="Margins hold at 34.1 [C:margin-hold].")
    clean_risk = _risk_payload(reasoning="Affirming at 34.1 margin [C:margin-hold] [RF00].")
    coro, _ = _run([clean_research, clean_risk])

    memo = await coro

    assert not any("34.1" in g for g in memo.data_gaps)


# ---------------------------------------------------------------------------
# The Research Manager issues no verdict — removed 2026-08-26 (code review):
# it was shown to the Judge as prior context and measured live (ASML) to
# flip-flop sell/hold/sell across production-temperature samples of the
# SAME fixed debate, a pure noise source with nothing downstream requiring
# it. The Risk Judge's `verdict` is now the pipeline's only verdict.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_the_risk_judges_verdict_is_the_memos_only_verdict():
    coro, _ = _run([_research_payload(), _risk_payload(verdict="sell")])

    memo = await coro

    assert memo.verdict == Verdict.SELL
    assert not hasattr(memo, "research_preliminary_verdict")


def test_the_risk_judge_tool_schema_never_offers_unresolved():
    """`Verdict.UNRESOLVED` (added 2026-08-26 alongside majority-of-N
    sampling in application/nodes.py) is a Python-computed aggregate over
    several Risk Judge calls — no single call should ever be ABLE to pick
    it. `RiskJudgePayload.verdict` uses the narrower `IndividualVerdict`
    type specifically so the tool schema sent to the model enforces this
    structurally, not by prompt instruction alone. Reading the actual
    schema `_risk_judge_tool()` sends, not just the Python type, since a
    stale `.model_json_schema()` cache or a schema-generation quirk would
    be exactly the kind of gap a type-only assertion misses."""
    schema = port._risk_judge_tool()["input_schema"]
    verdict_enum = schema["properties"]["verdict"]["enum"]

    assert "unresolved" not in verdict_enum
    assert set(verdict_enum) == {"buy", "sell", "hold"}


@pytest.mark.anyio
async def test_research_manager_never_sees_the_risk_ledger():
    """build_research_pack must not leak [RFnn] ids into what the Research
    Manager is shown — it's a structural guarantee (the pack function simply
    never calls _render_ledger), asserted here against the actual system
    prompt content sent on the first call."""
    coro, client = _run([_research_payload(), _risk_payload()])
    await coro

    research_call = client.messages.calls[0]
    system_text = "\n".join(b["text"] for b in research_call["system"])
    assert "RISK LEDGER" not in system_text


@pytest.mark.anyio
async def test_risk_judge_is_shown_the_research_managers_output():
    coro, client = _run([_research_payload(), _risk_payload()])
    await coro

    risk_judge_call = client.messages.calls[1]
    system_text = "\n".join(b["text"] for b in risk_judge_call["system"])
    assert "RESEARCH MANAGER'S SYNTHESIS" in system_text
    assert _research_payload()["thesis"] in system_text
    assert "Preliminary verdict" not in system_text   # the field no longer exists


# ---------------------------------------------------------------------------
# Happy path — memo assembly
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_happy_path_assembles_a_complete_memo():
    coro, _ = _run([_research_payload(), _risk_payload()])
    memo = await coro

    assert memo.ticker == "ACN"
    assert memo.verdict == Verdict.HOLD
    assert memo.data_as_of_date == date(2026, 8, 19)
    assert "a base gap" in memo.data_gaps
    assert "a base evidence line" in memo.evidence
    assert 0.0 <= memo.confidence <= 1.0
    assert memo.watch_items == _risk_payload()["watch_items"]
    assert memo.bull_case == _research_payload()["bull_case"]
    assert memo.research_thesis == _research_payload()["thesis"]


# ---------------------------------------------------------------------------
# Confidence — computed, not model-emitted
# ---------------------------------------------------------------------------

def test_confidence_is_bounded_and_penalizes_contestation_and_flags():
    full_coverage_uncontested = port.compute_confidence(
        _state(technical_report=object(), news_digest=object()),
        ledger=[], debate_turns=[],
    )
    heavily_contested = port.compute_confidence(
        _state(),
        ledger=[
            RiskLedgerEntry(
                factor_id="RF00", text="x", trigger="y", horizon="quarters",
                evidence_ref="none", evidence_quote="", proposed_by="neutral",
                scores={"aggressive": (5, 5), "conservative": (1, 1)},
                severity_spread=4, likelihood_spread=4, contested=True,
            )
        ],
        debate_turns=[],
    )
    assert 0.0 <= heavily_contested <= 1.0
    assert 0.0 <= full_coverage_uncontested <= 1.0
    assert heavily_contested < full_coverage_uncontested


def test_confidence_never_exceeds_bounds_regardless_of_inputs():
    assert 0.0 <= port.compute_confidence(_state(), ledger=[], debate_turns=[]) <= 1.0


# ---------------------------------------------------------------------------
# Temperature / model plumbing — the actual gap this rewrite closes
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_production_calls_omit_temperature_by_default():
    coro, client = _run([_research_payload(), _risk_payload()])
    await coro

    assert "temperature" not in client.messages.calls[0]
    assert "temperature" not in client.messages.calls[1]


@pytest.mark.anyio
async def test_an_explicit_temperature_is_threaded_to_both_calls_and_disables_thinking():
    client = _FakeClient([_research_payload(), _risk_payload()])
    state = _state()

    await port.run_synthesis(
        state, ledger=_ledger(), base_gaps=[], base_evidence=[], as_of=state["as_of_date"],
        client=client, research_temperature=0.0, risk_temperature=0.0,
    )

    for call in client.messages.calls:
        assert call["temperature"] == 0.0
        assert "thinking" not in call


def test_research_manager_and_risk_judge_follow_the_project_wide_model_by_default():
    """Not pinned to Sonnet, despite the spec text naming it — Sonnet 5
    deprecated `temperature` outright, which undercuts the determinism
    guarantee these two roles exist to support. See synthesis_port's module
    docstring and create_with_temperature_fallback."""
    from app.agent.researcher import AGENT_MODEL

    assert port.RESEARCH_MANAGER_MODEL == AGENT_MODEL
    assert port.RISK_JUDGE_MODEL == AGENT_MODEL
