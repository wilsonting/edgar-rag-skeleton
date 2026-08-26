"""Guardrail tests for the risk panel port. Mocked LLM throughout — no
network, no cost. Focused on the one thing Phase 6 exists to prove out: that
Python, not the model, owns the factor-id space (see domain/risk.py's module
docstring and docs/phase6-gate-a-findings.md for why).
"""

from __future__ import annotations

from datetime import date

import pytest

import app.agent.trading.infrastructure.risk_port as port
from app.agent.trading.domain.fundamentals_report import FundamentalsReport
from app.agent.trading.domain.risk import RiskFactor, RiskScore, RiskTurn, RiskTurnPayload


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


def _factor_payload(**overrides) -> dict:
    base = {
        "factor_id": "unassigned",
        "text": "Customer concentration risk.",
        "trigger": "top customer exceeds 25% of revenue in the next 10-K",
        "horizon": "quarters",
        "evidence_ref": "none",
        "evidence_quote": "none",
    }
    base.update(overrides)
    return base


def _turn_payload(**overrides) -> dict:
    base = {
        "argument": "Customer concentration is the dominant tail risk here.",
        "proposes": [],
        "scores": [],
        "accept_condition": "none",
    }
    base.update(overrides)
    return base


def _state(**extra) -> dict:
    state = {
        "ticker": "ACN",
        "fundamentals_report": FundamentalsReport(
            ticker="ACN", summary="Revenue grew.", input_tokens=0, cache_write_tokens=0,
            cache_read_tokens=0, output_tokens=0, generated_at=date(2026, 8, 19),
        ),
    }
    state.update(extra)
    return state


@pytest.fixture(autouse=True)
def _no_cost_log(monkeypatch):
    monkeypatch.setattr(port, "log_cost", lambda *a, **k: 0.02)


# ---------------------------------------------------------------------------
# factor_id is Python-assigned, content-addressed — NOT positional
# ---------------------------------------------------------------------------
#
# Positional ids (f"RF{i:02d}") were the original design and are gone: found
# live (2026-08-26, code review) that two temperature=0 replays of the
# IDENTICAL prompt produced enumerations that differed in count and had the
# same underlying concepts under swapped positional ids — "Python owns
# identity" was true of the label, not of what the label pointed at. See
# risk_port._content_id's docstring for the full finding.

@pytest.mark.anyio
async def test_enumeration_turn_gets_python_assigned_ids_regardless_of_what_the_model_sent():
    payload = _turn_payload(
        proposes=[
            _factor_payload(factor_id="whatever-i-felt-like", text="Risk A"),
            _factor_payload(factor_id="", text="Risk B"),
        ]
    )
    client = _FakeClient([payload])

    turn = await port.run_risk_turn(_state(), "neutral", 0, client=client)

    ids = [f.factor_id for f in turn.payload.proposes]
    assert all(fid.startswith("RF") and fid != "whatever-i-felt-like" and fid != "" for fid in ids)
    assert len(set(ids)) == 2   # distinct factors get distinct ids


@pytest.mark.anyio
async def test_the_same_factor_text_gets_the_same_id_across_independent_calls():
    """The actual property the content-addressed scheme exists to
    guarantee: identity survives a replay, which a positional counter
    never could — this is the direct regression test for the live finding."""
    payload_a = _turn_payload(proposes=[_factor_payload(text="Price breaks below the 200-day moving average")])
    payload_b = _turn_payload(proposes=[_factor_payload(text="Price breaks below the 200-day moving average")])

    turn_a = await port.run_risk_turn(_state(), "neutral", 0, client=_FakeClient([payload_a]))
    turn_b = await port.run_risk_turn(_state(), "neutral", 0, client=_FakeClient([payload_b]))

    assert turn_a.payload.proposes[0].factor_id == turn_b.payload.proposes[0].factor_id


@pytest.mark.anyio
async def test_wording_and_case_differences_still_hash_to_the_same_id():
    payload_a = _turn_payload(proposes=[_factor_payload(text="RSI falls below 30")])
    payload_b = _turn_payload(proposes=[_factor_payload(text="  RSI Falls Below 30!!  ")])

    turn_a = await port.run_risk_turn(_state(), "neutral", 0, client=_FakeClient([payload_a]))
    turn_b = await port.run_risk_turn(_state(), "neutral", 0, client=_FakeClient([payload_b]))

    assert turn_a.payload.proposes[0].factor_id == turn_b.payload.proposes[0].factor_id


@pytest.mark.anyio
async def test_different_factor_text_gets_different_ids():
    payload = _turn_payload(proposes=[
        _factor_payload(text="RSI falls below 30"),
        _factor_payload(text="Price breaks below the 200-day moving average"),
    ])
    client = _FakeClient([payload])

    turn = await port.run_risk_turn(_state(), "neutral", 0, client=client)

    ids = [f.factor_id for f in turn.payload.proposes]
    assert len(set(ids)) == 2


@pytest.mark.anyio
async def test_a_new_factor_does_not_collide_with_an_id_already_on_the_slate():
    prior = RiskTurn(
        turn_index=0, round_num=1, persona="neutral",
        payload=RiskTurnPayload(
            proposes=[RiskFactor(factor_id="RFAAAA", text="x", trigger="closes below 100", horizon="weeks", evidence_ref="none")],
            argument="a",
        ),
    )
    payload = _turn_payload(proposes=[_factor_payload(text="new one")])
    client = _FakeClient([payload])

    turn = await port.run_risk_turn(_state(risk_turns=[prior]), "aggressive", 1, client=client)

    new_id = turn.payload.proposes[0].factor_id
    assert new_id.startswith("RF")
    assert new_id != "RFAAAA"


# ---------------------------------------------------------------------------
# Proposal-count enforcement — truncation, not just a flag (the id-space
# integrity rule; see risk_port.py's guardrails-section docstring for why
# this one is structural rather than observational)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_a_scoring_turn_proposing_two_factors_keeps_only_the_first():
    payload = _turn_payload(
        scores=[],
        proposes=[_factor_payload(text="first"), _factor_payload(text="second")],
        accept_condition="price closes below 50",
    )
    client = _FakeClient([payload])

    turn = await port.run_risk_turn(_state(), "aggressive", 1, client=client)

    assert [f.text for f in turn.payload.proposes] == ["first"]
    assert any("over_proposal" in f for f in turn.guard_flags)


@pytest.mark.anyio
async def test_an_adjudication_turn_proposing_anything_gets_it_all_dropped():
    prior = [
        RiskTurn(turn_index=0, round_num=1, persona="neutral", payload=RiskTurnPayload(
            argument="a", proposes=[RiskFactor(factor_id="RF00", text="x", trigger="closes below 100", horizon="weeks", evidence_ref="none")])),
        RiskTurn(turn_index=1, round_num=1, persona="aggressive", payload=RiskTurnPayload(
            argument="a", scores=[RiskScore(factor_id="RF00", severity=5, likelihood=5, rationale="r")], accept_condition="x")),
        RiskTurn(turn_index=2, round_num=1, persona="conservative", payload=RiskTurnPayload(
            argument="a", scores=[RiskScore(factor_id="RF00", severity=1, likelihood=1, rationale="r")], accept_condition="x")),
    ]
    payload = _turn_payload(proposes=[_factor_payload(text="late")], scores=[])
    client = _FakeClient([payload])

    turn = await port.run_risk_turn(_state(risk_turns=prior), "neutral", 3, client=client)

    assert turn.payload.proposes == []
    assert any("late_proposal" in f for f in turn.guard_flags)


# ---------------------------------------------------------------------------
# Trigger falsifiability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "trigger,falsifiable",
    [
        ("closes below 850", True),
        ("Q3 2027 earnings miss consensus", True),
        ("exceeds 40% of revenue", True),
        ("if confidence erodes", False),
        ("sentiment turns negative", False),
    ],
)
def test_falsifiable_trigger_detection(trigger, falsifiable):
    assert port._is_falsifiable_trigger(trigger) is falsifiable


@pytest.mark.anyio
async def test_unfalsifiable_trigger_is_flagged_not_blocked():
    payload = _turn_payload(proposes=[_factor_payload(trigger="if sentiment sours")])
    client = _FakeClient([payload])

    turn = await port.run_risk_turn(_state(), "neutral", 0, client=client)

    assert any("unfalsifiable_trigger" in f for f in turn.guard_flags)
    assert len(turn.payload.proposes) == 1   # flagged, not dropped


# ---------------------------------------------------------------------------
# accept_condition
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_missing_accept_condition_on_a_scoring_turn_is_flagged():
    payload = _turn_payload(accept_condition="none")
    client = _FakeClient([payload])

    turn = await port.run_risk_turn(_state(), "aggressive", 1, client=client)

    assert "no_accept_condition" in turn.guard_flags


@pytest.mark.anyio
async def test_accept_condition_present_is_not_flagged():
    payload = _turn_payload(accept_condition="closes above 900 by year end")
    client = _FakeClient([payload])

    turn = await port.run_risk_turn(_state(), "aggressive", 1, client=client)

    assert "no_accept_condition" not in turn.guard_flags


@pytest.mark.anyio
async def test_accept_condition_is_not_required_on_the_enumeration_turn():
    payload = _turn_payload(accept_condition="none")
    client = _FakeClient([payload])

    turn = await port.run_risk_turn(_state(), "neutral", 0, client=client)

    assert "no_accept_condition" not in turn.guard_flags


# ---------------------------------------------------------------------------
# Slate completeness / unknown ids
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_slate_incomplete_is_flagged_when_a_scoring_turn_skips_an_id():
    prior = RiskTurn(turn_index=0, round_num=1, persona="neutral", payload=RiskTurnPayload(
        argument="a", proposes=[
            RiskFactor(factor_id="RF00", text="x", trigger="closes below 100", horizon="weeks", evidence_ref="none"),
            RiskFactor(factor_id="RF01", text="y", trigger="closes below 50", horizon="weeks", evidence_ref="none"),
        ]))
    payload = _turn_payload(
        scores=[{"factor_id": "RF00", "severity": 3, "likelihood": 3, "rationale": "r"}],
        accept_condition="x",
    )
    client = _FakeClient([payload])

    turn = await port.run_risk_turn(_state(risk_turns=[prior]), "aggressive", 1, client=client)

    assert any("slate_incomplete" in f and "RF01" in f for f in turn.guard_flags)


@pytest.mark.anyio
async def test_unknown_factor_id_in_a_score_is_flagged():
    prior = RiskTurn(turn_index=0, round_num=1, persona="neutral", payload=RiskTurnPayload(
        argument="a", proposes=[RiskFactor(factor_id="RF00", text="x", trigger="closes below 100", horizon="weeks", evidence_ref="none")]))
    payload = _turn_payload(
        scores=[
            {"factor_id": "RF00", "severity": 3, "likelihood": 3, "rationale": "r"},
            {"factor_id": "RF99", "severity": 3, "likelihood": 3, "rationale": "r"},
        ],
        accept_condition="x",
    )
    client = _FakeClient([payload])

    turn = await port.run_risk_turn(_state(risk_turns=[prior]), "aggressive", 1, client=client)

    assert any("unknown_factor_id" in f and "RF99" in f for f in turn.guard_flags)


# ---------------------------------------------------------------------------
# Numeric guard corpus includes the risk panel's OWN prior transcript
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_a_number_from_an_earlier_risk_turns_own_trigger_is_not_flagged():
    """Found live (MSFT, 2026-08-25): RF03's trigger ("RSI falls below 60"),
    proposed at turn 0, was flagged as unbacked when a later turn cited the
    same "60" — the numeric guard's corpus was reports + debate only, never
    the risk panel's own running transcript, even though every later turn is
    SHOWN that transcript in its prompt (render_risk_transcript)."""
    prior = RiskTurn(
        turn_index=0, round_num=1, persona="neutral",
        payload=RiskTurnPayload(
            argument="a",
            proposes=[RiskFactor(
                factor_id="RF03", text="x",
                trigger="RSI falls below 60 while MACD histogram remains negative",
                horizon="weeks", evidence_ref="none",
            )],
        ),
    )
    payload = _turn_payload(
        argument="The trigger requires RSI to fall below 60, which is plausible.",
        scores=[{"factor_id": "RF03", "severity": 3, "likelihood": 3, "rationale": "r"}],
        accept_condition="x",
    )
    client = _FakeClient([payload])

    turn = await port.run_risk_turn(_state(risk_turns=[prior]), "aggressive", 1, client=client)

    assert not any("unbacked_number: 60" in f for f in turn.guard_flags)


@pytest.mark.anyio
async def test_a_number_from_an_earlier_risk_turns_own_score_is_not_flagged():
    prior = RiskTurn(
        turn_index=0, round_num=1, persona="neutral",
        payload=RiskTurnPayload(argument="a", proposes=[
            RiskFactor(factor_id="RF00", text="x", trigger="closes below 100", horizon="weeks", evidence_ref="none"),
        ]),
    )
    scored = RiskTurn(
        turn_index=1, round_num=1, persona="aggressive",
        payload=RiskTurnPayload(
            argument="a",
            scores=[RiskScore(factor_id="RF00", severity=2, likelihood=3, rationale="r")],
            accept_condition="x",
        ),
    )
    payload = _turn_payload(
        argument="The aggressive panelist scored severity 2, likelihood 3 on this factor.",
        scores=[{"factor_id": "RF00", "severity": 3, "likelihood": 3, "rationale": "r"}],
        accept_condition="x",
    )
    client = _FakeClient([payload])

    turn = await port.run_risk_turn(_state(risk_turns=[prior, scored]), "conservative", 2, client=client)

    assert not any("unbacked_number: 2" in f for f in turn.guard_flags)


# ---------------------------------------------------------------------------
# turn_phase generalizes over RISK_MAX_ROUNDS (Phase 6 gap closure: 2 -> 3
# rounds) rather than being hardcoded to a fixed 6-turn shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "turn_index,expected",
    [
        (0, "enumerate"),
        (1, "score"), (2, "score"),
        (3, "adjudicate"), (4, "respond"), (5, "respond"),
        (6, "adjudicate"), (7, "respond"), (8, "respond"),
    ],
)
def test_turn_phase_over_three_rounds(turn_index, expected):
    assert port.turn_phase(turn_index) == expected


# ---------------------------------------------------------------------------
# Temperature plumbing — the determinism/stability check needs this
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_production_calls_omit_temperature_by_default():
    payload = _turn_payload()
    client = _FakeClient([payload])

    await port.run_risk_turn(_state(), "neutral", 0, client=client)

    assert "temperature" not in client.messages.calls[0]


@pytest.mark.anyio
async def test_an_explicit_temperature_is_sent_and_disables_thinking():
    payload = _turn_payload()
    client = _FakeClient([payload])

    await port.run_risk_turn(_state(), "neutral", 0, client=client, temperature=0.0)

    call = client.messages.calls[0]
    assert call["temperature"] == 0.0
    assert "thinking" not in call
