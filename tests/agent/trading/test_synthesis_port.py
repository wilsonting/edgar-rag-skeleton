"""Synthesis port tests — Phase 6 exit criteria 4 and 5.

Criterion 4: synthesis emits ZERO unresolvable references — an injected
[RF99] must raise, not silently drop. Criterion 5: synthesis emits ZERO
unbacked numbers — a number present in no report and no claim must be
flagged (blocked, if it lands in reasoning/risk_narrative specifically).

Mocked LLM throughout, same fake-client pattern as test_debate_port.py and
test_risk_port.py — no network, no cost.
"""

from __future__ import annotations

from datetime import date

import pytest

import app.agent.trading.infrastructure.synthesis_port as port
from app.agent.trading.domain.debate import DebateClaim, DebateTurn, DebateTurnPayload
from app.agent.trading.domain.decision_memo import SynthesisPayload, Verdict
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
    def __init__(self, payloads: list[dict | None]):
        self.messages = _FakeMessages(payloads)


def _payload(**overrides) -> dict:
    base = {
        "bull_case": "Margins are stable [C:margin-hold].",
        "bear_case": "Growth is slowing [C:margin-hold].",
        "risk_narrative": "Customer concentration is the key risk [RF00].",
        "reasoning": "The evidence on balance supports a hold [C:margin-hold] [RF00].",
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
    payload = SynthesisPayload(**_payload())
    refs = port.extract_refs(payload)
    assert "margin-hold" in refs
    assert "RF00" in refs


def test_resolve_refs_is_empty_when_every_citation_is_real():
    payload = SynthesisPayload(**_payload())
    claims = {c.claim_id: c for t in _debate_turns() for c in t.payload.claims}
    ledger_by_id = {e.factor_id: e for e in _ledger()}
    assert port.resolve_refs(payload, claims, ledger_by_id) == []


def test_resolve_refs_names_the_unresolved_id():
    payload = SynthesisPayload(**_payload(risk_narrative="A latent risk exists [RF99]."))
    claims = {c.claim_id: c for t in _debate_turns() for c in t.payload.claims}
    ledger_by_id = {e.factor_id: e for e in _ledger()}
    assert port.resolve_refs(payload, claims, ledger_by_id) == ["RF99"]


# ---------------------------------------------------------------------------
# Criterion 4 — reference integrity, through run_synthesis end to end
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_an_injected_unresolvable_reference_raises_not_drops():
    """The exit-criterion case: [RF99] does not exist in the ledger. Both
    attempts (initial + the one correction retry) still cite it, so the run
    must raise SynthesisReferenceError, not silently produce a memo with a
    dangling citation."""
    bad = _payload(risk_narrative="A latent risk exists [RF99].")
    coro, client = _run([bad, bad])

    with pytest.raises(port.SynthesisReferenceError, match="RF99"):
        await coro

    assert len(client.messages.calls) == 2   # exactly one retry attempt, not a loop


@pytest.mark.anyio
async def test_a_corrected_reference_on_retry_succeeds():
    bad = _payload(risk_narrative="A latent risk exists [RF99].")
    good = _payload()
    coro, _ = _run([bad, good])

    memo = await coro

    assert memo.verdict == Verdict.HOLD
    assert any("RF00" in e or "Customer concentration" in e for e in memo.evidence)


# ---------------------------------------------------------------------------
# Criterion 5 — fabrication guard
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_an_unbacked_number_in_reasoning_blocks_the_run():
    """A number present in NO report, NO debate claim, and NO risk factor,
    placed in `reasoning` — the load-bearing field — must block."""
    bait = _payload(reasoning="Revenue could reach $9,999 million by next year [C:margin-hold].")
    coro, _ = _run([bait])

    with pytest.raises(port.SynthesisFabricationError, match="9999|9,999"):
        await coro


@pytest.mark.anyio
async def test_an_unbacked_number_in_risk_narrative_blocks_the_run():
    bait = _payload(risk_narrative="Losses could hit $12,345 thousand [RF00].")
    coro, _ = _run([bait])

    with pytest.raises(port.SynthesisFabricationError):
        await coro


@pytest.mark.anyio
async def test_an_unbacked_number_in_bull_case_is_a_gap_not_a_block():
    """Elsewhere (bull/bear case, watch items) an unbacked number is a
    data_gaps entry, not a run-blocking error — the two-tier response
    documented in synthesis_port._numeric_guard."""
    bait = _payload(bull_case="Upside could reach $7,777 million [C:margin-hold].")
    coro, _ = _run([bait])

    memo = await coro

    assert any("7777" in g or "7,777" in g for g in memo.data_gaps)


@pytest.mark.anyio
async def test_a_faithfully_cited_number_from_the_pack_is_not_flagged():
    """34.1 appears verbatim in the fundamentals report — restating it
    should not trip either guard tier."""
    clean = _payload(reasoning="Margins hold at 34.1 [C:margin-hold] [RF00].")
    coro, _ = _run([clean])

    memo = await coro

    assert not any("34.1" in g for g in memo.data_gaps)


# ---------------------------------------------------------------------------
# Happy path — memo assembly
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_happy_path_assembles_a_complete_memo():
    coro, _ = _run([_payload()])
    memo = await coro

    assert memo.ticker == "ACN"
    assert memo.verdict == Verdict.HOLD
    assert memo.data_as_of_date == date(2026, 8, 19)
    assert "a base gap" in memo.data_gaps
    assert "a base evidence line" in memo.evidence
    assert 0.0 <= memo.confidence <= 1.0
    assert memo.watch_items == _payload()["watch_items"]


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
