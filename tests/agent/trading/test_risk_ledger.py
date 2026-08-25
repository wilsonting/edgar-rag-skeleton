"""Adversarial fixtures for build_risk_ledger — Phase 6 exit criterion 3.

Pure-function tests, no state, no I/O, no graph. The contract this guards
(domain/risk.py's RiskLedgerEntry docstring, risk_ledger.py's module
docstring) is specifically about what does NOT happen on a missing score:
no default, no average, no imputed value anywhere — silent imputation is the
single most plausible way this function could manufacture a number no
persona ever asserted.
"""

from __future__ import annotations

from app.agent.trading.application.risk_ledger import build_risk_ledger, contested_ids
from app.agent.trading.domain.risk import PERSONAS, RiskFactor, RiskScore, RiskTurn, RiskTurnPayload


def _factor(factor_id: str, text: str = "risk", **kw) -> RiskFactor:
    return RiskFactor(
        factor_id=factor_id,
        text=text,
        trigger=kw.pop("trigger", "price closes below 100 on 2026-12-31"),
        horizon=kw.pop("horizon", "quarters"),
        evidence_ref=kw.pop("evidence_ref", "none"),
        evidence_quote=kw.pop("evidence_quote", ""),
    )


def _turn(index: int, persona, proposes=(), scores=()) -> RiskTurn:
    return RiskTurn(
        turn_index=index,
        round_num=(index // len(PERSONAS)) + 1,
        persona=persona,
        payload=RiskTurnPayload(
            argument="stub",
            proposes=list(proposes),
            scores=list(scores),
        ),
    )


def test_factor_proposed_but_never_scored_is_present_with_all_personas_missing():
    turns = [_turn(0, "neutral", proposes=[_factor("RF00")])]

    ledger = build_risk_ledger(turns)

    assert len(ledger) == 1
    entry = ledger[0]
    assert entry.factor_id == "RF00"
    assert entry.scores == {}
    assert sorted(entry.missing_scores) == sorted(PERSONAS)
    assert entry.contested is False
    assert entry.severity_spread == 0
    assert entry.likelihood_spread == 0


def test_one_persona_omitting_a_factor_leaves_no_imputed_value():
    turns = [
        _turn(0, "neutral", proposes=[_factor("RF00")]),
        _turn(1, "aggressive", scores=[RiskScore(factor_id="RF00", severity=4, likelihood=3, rationale="r")]),
        _turn(2, "conservative", scores=[RiskScore(factor_id="RF00", severity=2, likelihood=3, rationale="r")]),
        # neutral never scores RF00 in this fixture
    ]

    ledger = build_risk_ledger(turns)

    entry = ledger[0]
    assert entry.missing_scores == ["neutral"]
    assert set(entry.scores) == {"aggressive", "conservative"}
    assert entry.scores["aggressive"] == (4, 3)
    assert entry.scores["conservative"] == (2, 3)
    # spread computed ONLY over the two present scores — no third value
    # anywhere, imputed or otherwise
    assert entry.severity_spread == 2
    assert entry.likelihood_spread == 0
    assert entry.contested is True   # severity_spread >= 2


def test_a_factor_with_a_single_score_has_zero_spread_but_is_not_contested_and_not_agreed():
    """Spread==0 here means 'nothing to compare', not 'agreement' — the
    caller must read missing_scores to tell the two apart."""
    turns = [
        _turn(0, "neutral", proposes=[_factor("RF00")]),
        _turn(1, "aggressive", scores=[RiskScore(factor_id="RF00", severity=5, likelihood=5, rationale="r")]),
    ]

    ledger = build_risk_ledger(turns)

    entry = ledger[0]
    assert entry.severity_spread == 0
    assert entry.likelihood_spread == 0
    assert entry.contested is False
    assert sorted(entry.missing_scores) == ["conservative", "neutral"]


def test_duplicate_score_for_one_factor_in_one_turn_keeps_the_first():
    turns = [
        _turn(0, "neutral", proposes=[_factor("RF00")]),
        _turn(
            1,
            "aggressive",
            scores=[
                RiskScore(factor_id="RF00", severity=5, likelihood=5, rationale="first"),
                RiskScore(factor_id="RF00", severity=1, likelihood=1, rationale="second"),
            ],
        ),
    ]

    ledger = build_risk_ledger(turns)

    assert ledger[0].scores["aggressive"] == (5, 5)


def test_score_for_an_id_absent_from_the_slate_is_dropped_not_created():
    turns = [
        _turn(0, "neutral", proposes=[_factor("RF00")]),
        _turn(1, "aggressive", scores=[RiskScore(factor_id="RF99", severity=3, likelihood=3, rationale="r")]),
    ]

    ledger = build_risk_ledger(turns)

    assert [e.factor_id for e in ledger] == ["RF00"]
    assert ledger[0].scores == {}   # RF99's score touched nothing


def test_a_duplicate_proposal_of_the_same_factor_id_keeps_the_first_and_is_one_row():
    turns = [
        _turn(0, "neutral", proposes=[_factor("RF00", text="original")]),
        _turn(1, "aggressive", proposes=[_factor("RF00", text="respoken")]),
    ]

    ledger = build_risk_ledger(turns)

    assert len(ledger) == 1
    assert ledger[0].text == "original"


def test_ledger_preserves_slate_order_across_multiple_factors():
    turns = [
        _turn(0, "neutral", proposes=[_factor("RF00"), _factor("RF01"), _factor("RF02")]),
    ]

    ledger = build_risk_ledger(turns)

    assert [e.factor_id for e in ledger] == ["RF00", "RF01", "RF02"]


def test_proposed_by_records_the_turn_that_first_introduced_the_factor():
    turns = [
        _turn(0, "neutral", proposes=[_factor("RF00")]),
        _turn(1, "aggressive", proposes=[_factor("RF01")]),
    ]

    ledger = build_risk_ledger(turns)

    assert {e.factor_id: e.proposed_by for e in ledger} == {
        "RF00": "neutral",
        "RF01": "aggressive",
    }


def test_contested_ids_returns_only_spread_at_or_above_threshold_in_slate_order():
    turns = [
        _turn(0, "neutral", proposes=[_factor("RF00"), _factor("RF01")]),
        _turn(1, "aggressive", scores=[
            RiskScore(factor_id="RF00", severity=5, likelihood=5, rationale="r"),
            RiskScore(factor_id="RF01", severity=3, likelihood=3, rationale="r"),
        ]),
        _turn(2, "conservative", scores=[
            RiskScore(factor_id="RF00", severity=1, likelihood=5, rationale="r"),
            RiskScore(factor_id="RF01", severity=3, likelihood=3, rationale="r"),
        ]),
    ]

    assert contested_ids(turns) == ["RF00"]


def test_empty_transcript_produces_an_empty_ledger():
    assert build_risk_ledger([]) == []
    assert contested_ids([]) == []
