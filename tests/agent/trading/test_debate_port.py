"""Guardrail tests. Mocked LLM throughout — no network, no cost.

The literature's consistent finding is that debate produces convergence and
convergence is not evidence of correctness: two instances of one base model
drift toward agreement because agreement is what the pretraining
distribution rewards. Four of the five counters below are enforced by
pydantic or Python precisely because a prompt-only guardrail is the kind that
degrades silently — and unlike the analyst nodes, NOTHING downstream of the
debate re-verifies its output.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import ValidationError

import app.agent.trading.infrastructure.debate_port as port
from app.agent.trading.domain.debate import (
    DebateClaim,
    DebateTurn,
    DebateTurnPayload,
    canonical_claims,
)
from app.agent.trading.domain.fundamentals_report import FundamentalsReport
from app.agent.trading.domain.news_digest import NewsDigest, NewsItem, SentimentSummary
from app.agent.trading.domain.technical_report import TechnicalIndicators, TechnicalReport


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

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
        "stance": "hold",
        "concession_trigger": "",
        "argument": "Margins hold up.",
        "claims": [
            {
                "claim_id": "margin-hold",
                "text": "Operating margin is stable.",
                "evidence_ref": "fundamentals",
                "evidence_quote": "operating margin of 34.1%",
            }
        ],
        "rebuts": [],
    }
    base.update(overrides)
    return base


def _state(**extra) -> dict:
    state = {
        "ticker": "ACN",
        "fundamentals_report": FundamentalsReport(
            ticker="ACN",
            summary="Revenue grew to 64.9 billion with an operating margin of 34.1%.",
            input_tokens=0,
            cache_write_tokens=0,
            cache_read_tokens=0,
            output_tokens=0,
            generated_at=date(2026, 8, 19),
        ),
    }
    state.update(extra)
    return state


def _turn(index: int, side: str, claim_ids: list[str], stance: str = "hold") -> DebateTurn:
    return DebateTurn(
        turn_index=index,
        round_num=(index // 2) + 1,
        side=side,
        payload=DebateTurnPayload(
            stance=stance,
            argument="prior",
            claims=[
                DebateClaim(claim_id=cid, text="t", evidence_ref="none")
                for cid in claim_ids
            ],
        ),
    )


@pytest.fixture(autouse=True)
def _no_cost_log(monkeypatch):
    """log_cost appends to docs/cost-log.jsonl. A test suite must not write
    into the real cost ledger — a run's spend has to stay auditable."""
    monkeypatch.setattr(port, "log_cost", lambda *a, **k: 0.02)


# ---------------------------------------------------------------------------
# (a) Concession must be structurally justified
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_concession_to_a_real_opposing_claim_is_accepted():
    turns = [_turn(0, "bull", ["margin-hold"])]
    client = _FakeClient(
        [_payload(stance="concede", concession_trigger="margin-hold", claims=[
            {"claim_id": "new", "text": "Fair.", "evidence_ref": "none"}
        ])]
    )

    turn = await port.run_debate_turn(
        _state(debate_turns=turns), "bear", 1, client=client
    )

    assert turn.payload.stance == "concede"
    assert turn.payload.concession_trigger == "margin-hold"


@pytest.mark.anyio
async def test_concession_to_a_claim_nobody_made_raises():
    """Makes "you know, that's a fair point" structurally impossible unless
    the fair point exists. Highest-value guardrail in the phase, and it costs
    nothing at runtime."""
    turns = [_turn(0, "bull", ["margin-hold"])]
    client = _FakeClient([_payload(stance="concede", concession_trigger="invented-id")])

    with pytest.raises(ValueError, match="not an opposing claim_id"):
        await port.run_debate_turn(_state(debate_turns=turns), "bear", 1, client=client)


@pytest.mark.anyio
async def test_conceding_to_your_own_earlier_claim_raises():
    """Agreeing with yourself is not a concession."""
    turns = [_turn(0, "bull", ["margin-hold"]), _turn(1, "bear", ["margin-soft"])]
    client = _FakeClient([_payload(stance="concede", concession_trigger="margin-hold")])

    with pytest.raises(ValueError, match="not an opposing claim_id"):
        await port.run_debate_turn(_state(debate_turns=turns), "bull", 2, client=client)


@pytest.mark.anyio
async def test_concession_trigger_on_a_non_concede_stance_raises():
    client = _FakeClient([_payload(stance="hold", concession_trigger="margin-hold")])

    with pytest.raises(ValueError, match="non-concede stance"):
        await port.run_debate_turn(_state(), "bull", 0, client=client)


# ---------------------------------------------------------------------------
# (a2) rebuts must point at real opposing claims — validation gap closed
# 2026-08-24. check_concession existed from day one; check_rebuts did not,
# so a turn could name a hallucinated or own-side id in `rebuts` and every
# guard would still pass. Measured across all five termination-run
# transcripts before this check existed: 95 of 95 rebutted ids resolved to a
# claim made in the immediately preceding turn — so on that evidence the gap
# was never exploited. It is closed as a completeness fix, not because a live
# run showed a hallucination.
# ---------------------------------------------------------------------------

def test_rebutting_a_real_opposing_claim_is_accepted():
    turns = [_turn(0, "bull", ["margin-hold"])]
    port.check_rebuts(
        DebateTurnPayload.model_validate(_payload(rebuts=["margin-hold"])),
        turns,
        "bear",
    )   # does not raise


def test_rebutting_a_claim_nobody_made_raises():
    with pytest.raises(ValueError, match="hallucinated id or the debater's own side"):
        port.check_rebuts(
            DebateTurnPayload.model_validate(_payload(rebuts=["invented-id"])),
            [],
            "bear",
        )


def test_rebutting_your_own_earlier_claim_raises():
    """A rebuttal names an OPPONENT's claim. Naming your own is not a
    rebuttal, and check_concession draws the same side/opposing-side line for
    the same reason."""
    turns = [_turn(0, "bull", ["own-claim"])]
    with pytest.raises(ValueError, match="own side"):
        port.check_rebuts(
            DebateTurnPayload.model_validate(_payload(rebuts=["own-claim"])),
            turns,
            "bull",
        )


def test_rebutting_an_older_opposing_claim_is_still_accepted():
    """Scoped like check_concession, not to the immediately preceding turn
    only: a later round legitimately returns to an earlier claim, and that
    must not be an error even though every rebuttal observed so far pointed
    at the previous turn specifically."""
    turns = [
        _turn(0, "bull", ["early-claim"]),
        _turn(1, "bear", ["bear-claim"]),
    ]
    port.check_rebuts(
        DebateTurnPayload.model_validate(_payload(rebuts=["early-claim"])),
        turns,
        "bear",
    )   # does not raise


def test_one_hallucinated_id_among_real_ones_still_raises():
    turns = [_turn(0, "bull", ["real-id"])]
    with pytest.raises(ValueError, match=r"\['fake-id'\]"):
        port.check_rebuts(
            DebateTurnPayload.model_validate(
                _payload(rebuts=["real-id", "fake-id"])
            ),
            turns,
            "bear",
        )


@pytest.mark.anyio
async def test_run_debate_turn_raises_on_a_hallucinated_rebuttal():
    """The end-to-end path, not just the unit function: a turn whose payload
    names a rebuttal id that does not exist must fail the node, the same way
    an unjustified concession does."""
    turns = [_turn(0, "bull", ["real-id"])]
    client = _FakeClient([_payload(rebuts=["invented-id"])])

    with pytest.raises(ValueError, match="hallucinated id"):
        await port.run_debate_turn(
            _state(debate_turns=turns), "bear", 1, client=client
        )


# ---------------------------------------------------------------------------
# (b) Numeric fabrication guard
# ---------------------------------------------------------------------------

PACK = (
    "FUNDAMENTALS:\nRevenue grew to 64.9 billion, operating margin 34.1%.\n"
    "TECHNICAL: last_close 327.55, volume_vs_20d_avg 0.529, rsi_14 41.2033."
)


def test_a_figure_absent_from_the_pack_is_flagged():
    assert port._flag_debate_numbers("Revenue reached 71.4 billion.", PACK) == ["71.4"]


def test_a_rounded_restatement_of_a_pack_value_is_not_flagged():
    """Prose rounds figures. Both of the first two live turns produced one of
    these — "RSI of 41.2" for 41.2033, "the 50-day at 330.12" for 330.1245 —
    a 100% false-positive rate on day one. A guard that cries wolf teaches
    the reader to skip it, and then it is not there for the one that
    matters."""
    assert port._flag_debate_numbers("RSI of 41.2 is neutral.", PACK) == []
    assert port._flag_debate_numbers("The 50-day sits at 327.6.", PACK) == []


def test_rounding_clearance_is_scoped_to_the_figure_s_own_precision():
    """Not a tolerance band. "41.2" clears only against a value in
    [41.15, 41.25), so the blind spot does not widen as the pack grows —
    which is the whole reason containment replaced tolerance here."""
    assert port._flag_debate_numbers("RSI of 41.3 is neutral.", PACK) == ["41.3"]
    assert port._flag_debate_numbers("RSI of 41.21 is neutral.", PACK) == ["41.21"]


def test_a_percent_written_against_a_percent_in_the_pack_is_not_flagged():
    """The pack states "operating margin 34.1%"; the debater writes "34%".
    Neither containment nor the ratio transform clears that — rounding
    does."""
    assert port._flag_debate_numbers("A margin of 34%.", PACK) == []


def test_a_figure_present_in_the_pack_is_not_flagged():
    assert port._flag_debate_numbers("Revenue of 64.9 billion.", PACK) == []


def test_a_faithful_percent_restatement_of_a_ratio_is_not_flagged():
    """volume_vs_20d_avg 0.529 reported as "53%" is faithful; containment
    alone would flag it, which is why percent forms get the Phase 3
    transforms."""
    assert port._flag_debate_numbers("Volume sits at 53% of its average.", PACK) == []


def test_a_fabricated_percent_is_still_flagged():
    assert port._flag_debate_numbers("Volume sits at 88% of its average.", PACK) == ["88%"]


@pytest.mark.parametrize(
    "text",
    [
        "downside is unconstrained until the low-30s oversold zone",
        "negative MACD and sub-50-SMA price action",
        "the post-2020 build-out",
    ],
)
def test_a_hyphenated_compound_is_not_read_as_a_negative_number(text):
    """Found on the first live debate (AVGO, 2026-08-23): "low-30s" and
    "sub-50-SMA" were reported as the fabricated figures -30 and -50 — two of
    that run's six flags, so it is not a rare shape.

    The lookbehind has to cover the digit as well as the sign. Blocking only
    "<letter>-<digits>" would let the scanner start one character later and
    flag a bare "30" out of "low-30s" — the same false positive with the sign
    filed off.
    """
    assert port._flag_debate_numbers(text, PACK) == []


def test_a_range_endpoint_is_still_read_as_a_positive_number():
    """The other half of the same lookbehind, from Phase 3: a hyphen between
    two digits is a range separator, and both endpoints are real values."""
    pack = "TECHNICAL: bb_lower 353.79, bb_upper 438.34."
    assert port._flag_debate_numbers("the band runs 353.79-438.34", pack) == []
    assert port._flag_debate_numbers("the band runs 353.79-999.99", pack) == ["999.99"]


def test_a_genuine_negative_number_still_parses():
    pack = "TECHNICAL: macd_histogram -5.8513."
    assert port._flag_debate_numbers("a histogram of -5.85", pack) == []
    assert port._flag_debate_numbers("a histogram of -9.99", pack) == ["-9.99"]


def test_period_labels_are_not_treated_as_data():
    assert port._flag_debate_numbers("Above its 200-day average.", PACK) == []


def test_typographic_minus_is_normalized_before_the_sign_is_read():
    """The model writes U+2212 roughly one run in four. float() agrees on the
    narrower alphabet, so an unnormalized minus parses as a positive number
    and matches nothing."""
    pack = "TECHNICAL: macd_histogram -1.2158."
    assert port._flag_debate_numbers("The histogram is −1.2158.", pack) == []


def test_the_same_fabricated_figure_repeated_is_one_finding():
    text = "71.4 billion, up from 71.4 the prior year, so 71.4 stands."
    assert port._flag_debate_numbers(text, PACK) == ["71.4"]


@pytest.mark.anyio
async def test_fabricated_numbers_reach_the_turn_record(monkeypatch):
    client = _FakeClient(
        [_payload(argument="Revenue reached 71.4 billion this year.")]
    )

    turn = await port.run_debate_turn(_state(), "bull", 0, client=client)

    assert "71.4" in turn.guard_flags


@pytest.mark.anyio
async def test_flagged_numbers_reach_the_memo_data_gaps():
    """The guard is only worth having if the finding travels. Nothing
    downstream of the debate re-verifies these figures — the memo caveat is
    the last stop."""
    from app.agent.trading.application.nodes import _debate_caveats

    turn = _turn(0, "bull", ["c0"])
    turn.guard_flags = ["71.4"]

    gaps, _ = _debate_caveats({"debate_turns": [turn]})

    assert any("71.4" in gap for gap in gaps)


def test_a_figure_repeated_across_turns_is_listed_once_with_a_count():
    """A figure the debate leans on gets restated every round. On the first
    live run that filled the whole display budget with "0.15, 0.15, 0.15"
    while two other flagged figures were hidden behind "(+1 more)" — the
    repetition told the reader nothing and cost them the part that would."""
    from app.agent.trading.application.nodes import _debate_caveats

    turns = []
    for i in range(4):
        turn = _turn(i, "bull" if i % 2 == 0 else "bear", [f"c{i}"])
        turn.guard_flags = ["0.15"] if i < 3 else ["0.15", "88.8"]
        turns.append(turn)

    gaps, _ = _debate_caveats({"debate_turns": turns})
    gap = next(g for g in gaps if "may be fabricated" in g)

    assert "0.15 (x4)" in gap
    assert "88.8" in gap
    assert gap.count("0.15") == 1
    assert "5 mention(s) of 2 figure(s)" in gap


# ---------------------------------------------------------------------------
# (c) Symmetry enforced in code
# ---------------------------------------------------------------------------

def test_the_two_system_prompts_differ_only_by_their_stance_paragraph():
    """As a CI test, so it fails when someone edits one prompt. Any asymmetry
    becomes a permanent confound in every transcript reasoned over later, and
    it goes invisible after a few prompt edits."""
    assert port.BULL_SYSTEM.replace(port.BULL_STANCE, "") == port.BEAR_SYSTEM.replace(
        port.BEAR_STANCE, ""
    )
    assert port.BULL_SYSTEM != port.BEAR_SYSTEM


@pytest.mark.anyio
async def test_both_sides_get_identical_call_parameters():
    """Symmetry is a guardrail, not tidiness: identical max_tokens, identical
    evidence pack, identical tool, one differing system block."""
    state = _state()
    bull_client = _FakeClient([_payload()])
    bear_client = _FakeClient([_payload()])

    await port.run_debate_turn(state, "bull", 0, client=bull_client)
    await port.run_debate_turn(
        {**state, "debate_turns": [_turn(0, "bull", ["margin-hold"])]},
        "bear",
        1,
        client=bear_client,
    )

    bull_call, bear_call = bull_client.messages.calls[0], bear_client.messages.calls[0]
    assert bull_call["max_tokens"] == bear_call["max_tokens"]
    assert bull_call["model"] == bear_call["model"]
    assert bull_call["tools"] == bear_call["tools"]
    assert bull_call["tool_choice"] == bear_call["tool_choice"]
    # same pack, cached the same way; only the stance block differs
    assert bull_call["system"][1] == bear_call["system"][1]
    assert bull_call["system"][1]["cache_control"] == {"type": "ephemeral"}
    assert bull_call["system"][0] != bear_call["system"][0]


# ---------------------------------------------------------------------------
# (d) Productivity
# ---------------------------------------------------------------------------

def test_a_turn_with_a_new_claim_id_is_productive():
    payload = DebateTurnPayload.model_validate(_payload())
    assert port.is_productive(payload, []) is True


def test_a_turn_that_only_restates_known_claim_ids_is_unproductive():
    payload = DebateTurnPayload.model_validate(_payload())
    prior = [_turn(0, "bear", ["margin-hold"])]
    assert port.is_productive(payload, prior) is False


# ---------------------------------------------------------------------------
# claim_id text stability — a completeness gap closed 2026-08-24.
#
# claim_id reuse is what the productivity lever and any future aggregation
# by id both depend on, and nothing checked that a reused id kept the same
# meaning. Found live (ACN, technical-only pack, 2026-08-24):
# `acn-volume-deteriorating` carried "collapsing conviction that exposes
# recovery moves to reversal risk" in turn 3 and "deteriorating participation
# that undermines recovery conviction" in turn 5 — one id, two claims.
# ---------------------------------------------------------------------------

def test_reusing_an_id_with_identical_text_is_not_flagged():
    turns = [_turn(0, "bull", ["stable-id"])]   # _turn's default claim text is "t"
    payload = DebateTurnPayload.model_validate(
        _payload(claims=[{"claim_id": "stable-id", "text": "t", "evidence_ref": "none"}])
    )
    assert port.check_claim_stability(payload, turns) == []


def test_reusing_an_id_with_different_text_is_flagged():
    """Flags, does not raise — see the module docstring on check_claim_
    stability for why a hard reject would make claim_id reuse impractical."""
    turns = [_turn(0, "bull", ["acn-volume-deteriorating"])]
    turns[0].payload.claims[0].text = (
        "collapsing conviction that exposes recovery moves to reversal risk"
    )
    payload = DebateTurnPayload.model_validate(
        _payload(claims=[{
            "claim_id": "acn-volume-deteriorating",
            "text": "deteriorating participation that undermines recovery conviction",
            "evidence_ref": "none",
        }])
    )
    assert port.check_claim_stability(payload, turns) == ["acn-volume-deteriorating"]


def test_stability_compares_against_the_first_occurrence_not_the_latest():
    """A third occurrence must be compared to the ORIGINAL text, not
    whatever the second occurrence drifted to — otherwise a slow drift across
    many turns could pass one check at a time while ending up far from what
    the id originally named."""
    t0 = _turn(0, "bull", ["drift-id"])
    t0.payload.claims[0].text = "original wording"
    t1 = _turn(1, "bear", ["drift-id"])
    t1.payload.claims[0].text = "already drifted once"   # 2nd occurrence

    payload = DebateTurnPayload.model_validate(
        _payload(claims=[{
            "claim_id": "drift-id",
            "text": "already drifted once",   # matches t1, not t0
            "evidence_ref": "none",
        }])
    )
    assert port.check_claim_stability(payload, [t0, t1]) == ["drift-id"]


def test_a_brand_new_claim_id_is_never_flagged():
    assert port.check_claim_stability(
        DebateTurnPayload.model_validate(_payload()), []
    ) == []


@pytest.mark.anyio
async def test_run_debate_turn_records_drift_without_raising():
    """The end-to-end path: drift lands on the turn record, not as an
    exception. A structural violation (check_concession, check_rebuts) kills
    the node; paraphrase drift is informational and must not."""
    turns = [_turn(0, "bull", ["shared-id"])]
    turns[0].payload.claims[0].text = "first wording"
    client = _FakeClient([_payload(claims=[{
        "claim_id": "shared-id",
        "text": "second wording",
        "evidence_ref": "none",
    }])])

    turn = await port.run_debate_turn(_state(debate_turns=turns), "bear", 1, client=client)

    assert turn.claim_text_drift == ["shared-id"]


# ---------------------------------------------------------------------------
# canonical_claims — the actual aggregation-safety mechanism. Lives in
# domain/debate.py, not here, because it has zero infra dependencies and
# Phase 6's risk debate is meant to import it directly.
# ---------------------------------------------------------------------------

def test_canonical_claims_keeps_the_first_occurrence_on_reuse():
    t0 = _turn(0, "bull", ["shared-id"])
    t0.payload.claims[0].text = "first wording"
    t1 = _turn(1, "bear", ["shared-id"])
    t1.payload.claims[0].text = "second wording"

    result = canonical_claims([t0, t1])

    assert result["shared-id"].text == "first wording"


def test_canonical_claims_covers_every_id_across_every_turn():
    t0 = _turn(0, "bull", ["a", "b"])
    t1 = _turn(1, "bear", ["c"])

    result = canonical_claims([t0, t1])

    assert set(result) == {"a", "b", "c"}


def test_canonical_claims_of_an_empty_transcript_is_empty():
    assert canonical_claims([]) == {}


# ---------------------------------------------------------------------------
# (e) Quote verification
# ---------------------------------------------------------------------------

def test_a_quote_present_in_the_named_report_verifies():
    payload = DebateTurnPayload.model_validate(_payload())
    assert port.check_quotes(payload, port.report_texts(_state())) == []


def test_a_quote_absent_from_the_named_report_is_recorded():
    payload = DebateTurnPayload.model_validate(
        _payload(claims=[{
            "claim_id": "margin-hold",
            "text": "Margin is stable.",
            "evidence_ref": "fundamentals",
            "evidence_quote": "operating margin of 51.9%",
        }])
    )
    assert port.check_quotes(payload, port.report_texts(_state())) == ["margin-hold"]


def test_quote_matching_tolerates_whitespace_differences():
    payload = DebateTurnPayload.model_validate(
        _payload(claims=[{
            "claim_id": "margin-hold",
            "text": "Margin is stable.",
            "evidence_ref": "fundamentals",
            "evidence_quote": "Operating   Margin\nof 34.1%",
        }])
    )
    assert port.check_quotes(payload, port.report_texts(_state())) == []


def _technical_state() -> dict:
    return _state(
        technical_report=TechnicalReport(
            ticker="ACN",
            as_of_date=date(2026, 8, 21),
            data_source="yfinance",
            bars_used=251,
            indicators=TechnicalIndicators(
                sma_200=368.2967127990723,
                rsi_14=38.721899422317186,
                last_close=368.45001220703125,
            ),
            interpretation="ACN is below its 50-day but above its 200-day average.",
        )
    )


def _quote_flags(quote: str, ref: str = "technical") -> list[str]:
    """Uses `quotable_texts`, the corpus `run_debate_turn` actually validates
    against in production — not `report_texts`, which still carries the raw
    JSON for `build_evidence_pack`/the number-fabrication guard. See
    `quotable_texts`'s docstring for why the two diverge."""
    payload = DebateTurnPayload.model_validate(
        _payload(claims=[{
            "claim_id": "c",
            "text": "t",
            "evidence_ref": ref,
            "evidence_quote": quote,
        }])
    )
    return port.check_quotes(payload, port.quotable_texts(_technical_state()))


def test_raw_json_is_not_a_valid_quote_source_regardless_of_punctuation():
    """This test's history is the arc of the underlying bug.

    Originally: the technical section was compact JSON, so the report read
    `"rsi_14":38.72` while a debater naturally wrote `rsi_14: 38.72` — same
    value, different punctuation. Comparing raw, that flagged 4 of 4 live
    Haiku claims, every one faithful. `_norm` was added to strip punctuation
    and whitespace before comparing, which fixed it — the assertion here used
    to be `== []` for both spellings.

    That fix was too permissive in a different way: `_norm` doesn't care
    WHAT text it matches against, so once it tolerated `rsi_14: 38.72` it
    also tolerated `"rsi_14":38.72` verbatim — the debater grepping the raw
    serialized dict rather than citing anything the analyst said. Found live
    (ACN, technical-only pack, 2026-08-24): 2 of 4 claims in one turn cited a
    raw `key":value` fragment this way. `evidence_ref != 'none'` is supposed
    to mean "grounded in a report", and a JSON fragment is not what the
    report SAYS.

    The actual fix is `quotable_texts` excluding the raw JSON from the
    corpus entirely (see `_render_technical`'s `quotable` param), so both
    spellings below now fail regardless of how well punctuation is
    normalized — normalization was never the right lever for this one.
    """
    assert _quote_flags("rsi_14: 38.721899422317186") == ["c"]
    assert _quote_flags('"rsi_14":38.721899422317186') == ["c"]


def test_the_relations_block_remains_quotable_after_the_json_exclusion():
    """The fix must not overcorrect: `derive_relations`'s output is Python-
    computed and authoritative (§8 of the as-built guide) — excluding raw
    JSON must not also exclude the relations sentences that were added
    specifically to give claims something legitimate to cite."""
    assert _quote_flags(
        "RSI (38.72) is NEITHER overbought nor oversold (between 30 and 70)"
    ) == []
    assert _quote_flags(
        "last close (368.45) is ABOVE the 200-day average (368.30)"
    ) == []


def test_a_quote_copied_across_a_line_wrap_verifies():
    assert _quote_flags(
        "ACN is below its 50-day but\n    above its 200-day average."
    ) == []


def test_a_span_spliced_from_two_non_adjacent_fields_is_still_flagged():
    """The normalization must not buy the false positives back by letting a
    fabricated contiguity through. Both live models did this — Sonnet with an
    ellipsis, Haiku with a comma — and it is a real finding: nothing in the
    report puts those two values side by side."""
    assert _quote_flags(
        "sma_200: 368.2967127990723, last_close: 368.45001220703125"
    ) == ["c"]
    assert _quote_flags(
        'sma_200":368.2967127990723... last_close":368.45001220703125'
    ) == ["c"]


def test_a_shifted_decimal_is_still_flagged():
    """Whitespace and quote marks are formatting; a decimal point is not.
    Stripping punctuation wholesale would let 3.872 pass as 38.72."""
    assert _quote_flags("rsi_14: 3.8721899422317186") == ["c"]
    assert _quote_flags("rsi_14: 55.0") == ["c"]


def test_a_claim_citing_no_report_is_not_quote_checked():
    payload = DebateTurnPayload.model_validate(
        _payload(claims=[{
            "claim_id": "derived",
            "text": "Both of those together imply pressure.",
            "evidence_ref": "none",
        }])
    )
    assert port.check_quotes(payload, port.report_texts(_state())) == []


# ---------------------------------------------------------------------------
# Schema violations and the one retry
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_an_empty_claims_list_is_retried_once_and_then_succeeds():
    client = _FakeClient([_payload(claims=[]), _payload()])

    turn = await port.run_debate_turn(_state(), "bull", 0, client=client)

    assert len(client.messages.calls) == 2
    assert turn.payload.claims[0].claim_id == "margin-hold"

    # The correction rides back as a tool_result, not a plain user turn. A
    # tool_use block must be answered by a tool_result in the very next
    # message; a text turn there is a 400, which is how this was found.
    last = client.messages.calls[1]["messages"][-1]
    assert last["role"] == "user"
    assert last["content"][0]["type"] == "tool_result"
    assert last["content"][0]["tool_use_id"] == "toolu_stub"
    assert last["content"][0]["is_error"] is True
    assert "did not validate" in last["content"][0]["content"]


@pytest.mark.anyio
async def test_a_second_schema_violation_raises_out_of_the_node():
    """Exactly one retry. Retries inside a node are invisible to the
    checkpointer, so an unbounded retry loop is a runaway the round cap
    cannot see — it lives inside a single super-step."""
    client = _FakeClient([_payload(claims=[]), _payload(claims=[])])

    with pytest.raises(ValidationError):
        await port.run_debate_turn(_state(), "bull", 0, client=client)

    assert len(client.messages.calls) == 2


@pytest.mark.anyio
async def test_a_response_with_no_tool_call_is_treated_as_a_schema_violation():
    client = _FakeClient([None, _payload()])

    turn = await port.run_debate_turn(_state(), "bull", 0, client=client)

    assert len(client.messages.calls) == 2
    assert turn.turn_index == 0
    # nothing to answer with a tool_result, so the correction is a plain turn
    assert client.messages.calls[1]["messages"][-1]["content"].startswith(
        "That submission did not validate"
    )


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

def test_the_tool_schema_carries_no_refs():
    """$ref resolution inside a tool input_schema is not something this code
    relies on; the nested DebateClaim is spliced in before sending."""
    blob = json.dumps(port.SUBMIT_TOOL["input_schema"])
    assert "$ref" not in blob
    assert "$defs" not in blob
    assert "claim_id" in blob


def test_the_tool_schema_is_strict_and_leaves_no_field_optional():
    """`strict: true` is what stopped the model flattening the payload — the
    DebateClaim fields hoisted to the top level with `stance` missing, on 3
    of 3 live turns. It costs the count bounds (strict rejects
    minItems/maxItems) and it forbids optional fields, which is why the
    'none' sentinel exists."""
    schema = port.SUBMIT_TOOL["input_schema"]
    claims = schema["properties"]["claims"]

    assert port.SUBMIT_TOOL["strict"] is True
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert claims["items"]["additionalProperties"] is False
    assert set(claims["items"]["required"]) == {
        "claim_id",
        "text",
        "evidence_ref",
        "evidence_quote",
    }
    # dropped from the wire schema, still enforced by pydantic on the way in
    assert "minItems" not in claims and "maxItems" not in claims
    assert "default" not in schema["properties"]["concession_trigger"]


def test_the_claim_count_bound_survives_in_pydantic_and_in_prose():
    """The bound the schema can no longer carry has to reach the model some
    other way, or it is not a bound at all."""
    field = DebateTurnPayload.model_fields["claims"]
    assert "1 and 5" in field.description
    assert "1 and 5" in json.dumps(port.SUBMIT_TOOL["input_schema"])

    with pytest.raises(ValidationError):
        DebateTurnPayload.model_validate(_payload(claims=[]))


def test_the_payload_schema_exposes_no_index_counter_or_side():
    """Python owns every index, counter and side label. If the model emitted
    its own round_num, the termination guard would be reading a field the
    model can fabricate."""
    fields = set(DebateTurnPayload.model_fields)
    assert fields.isdisjoint({"turn_index", "round_num", "side", "productive"})


# ---------------------------------------------------------------------------
# Evidence pack
# ---------------------------------------------------------------------------

def test_a_missing_analyst_leg_is_stated_not_omitted():
    """`--only news` leaves fundamentals as None. A debater reads silence as
    neutrality, which is the exact error the news caveats prevent one layer
    up."""
    pack = port.build_evidence_pack({"ticker": "ACN"})

    for name in ("fundamentals", "technical", "news", "sentiment"):
        assert f"{name.upper()}: NOT RUN" in pack
        assert f"Do not infer anything about {name}" in pack


def test_the_pack_states_the_computed_relations_authoritatively():
    """Phase 3 computes these comparisons in Python precisely because a model
    asked to work them out from raw values gets them wrong. The pack used to
    hand over the JSON and nothing else, and on the first live Haiku turns
    BOTH sides called an RSI of 38.7 "oversold" — every number real, so the
    numeric guard had nothing to catch."""
    pack = port.build_evidence_pack(_technical_state())

    assert "Computed relations (AUTHORITATIVE" in pack
    assert "RSI (38.72) is NEITHER overbought nor oversold (between 30 and 70)" in pack
    assert "last close (368.45) is ABOVE the 200-day average (368.30)" in pack
    # full precision is still there for a claim that turns on the 4th decimal
    assert "38.721899422317186" in pack


def test_the_system_prompt_forbids_contradicting_the_relations():
    """Whitespace-normalized, so reflowing the paragraph doesn't fail the
    suite — the rule has to be there, its line breaks don't."""
    for prompt in (port.BULL_SYSTEM, port.BEAR_SYSTEM):
        flat = " ".join(prompt.split())
        assert "AUTHORITATIVE" in flat
        assert "never contradict the block" in flat
        assert "whether RSI is overbought or oversold" in flat


def test_a_relation_line_is_quotable_as_a_single_span():
    """The other half of the fix. `evidence_quote` is one contiguous span, so
    a trend claim resting on two values that sit far apart in the JSON was a
    splice and got flagged. One relation line carries both values AND the
    comparison between them."""
    assert _quote_flags(
        "last close (368.45) is ABOVE the 200-day average (368.30)"
    ) == []
    assert _quote_flags(
        "RSI (38.72) is NEITHER overbought nor oversold (between 30 and 70)"
    ) == []


def test_a_fabricated_relation_is_still_flagged():
    """The relations block must not become a place where anything passes:
    inverting the comparison is exactly the claim the block exists to
    prevent, and it has to fail the quote check."""
    assert _quote_flags(
        "last close (368.45) is BELOW the 200-day average (368.30)"
    ) == ["c"]


def test_relations_degrade_when_indicators_are_missing():
    """A short history computes last_close and little else. The block should
    shrink, not raise — a partial report is a legitimate state."""
    state = _state(
        technical_report=TechnicalReport(
            ticker="ACN",
            as_of_date=date(2026, 8, 21),
            data_source="yfinance",
            bars_used=12,
            indicators=TechnicalIndicators(last_close=327.55),
            interpretation="Too little history for most indicators.",
        )
    )

    pack = port.build_evidence_pack(state)

    assert "Computed relations" in pack
    assert "overbought" not in pack
    assert "Too little history" in pack


def _digest_with_mix() -> NewsDigest:
    def item(headline, relevance):
        return NewsItem(
            headline=headline,
            published_date=AS_OF_NEWS,
            source="wire",
            url="https://example.com/x",
            summary=f"summary of {headline}",
            sentiment="positive" if relevance == "primary" else "neutral",
            relevance=relevance,
        )

    return NewsDigest(
        ticker="ACN",
        as_of_date=AS_OF_NEWS,
        window_start=date(2026, 8, 6),
        items=[
            item("Acme beats estimates", "primary"),
            item("Sector roundup names Acme", "mentioned"),
            item("Unrelated company files", "unrelated"),
            item("Acme raises guidance", "primary"),
        ],
        raw_article_count=9,
        deduped_count=4,
        dropped_out_of_window=5,
        dropped_missing_date=0,
        truncated_by_cap=False,
    )


AS_OF_NEWS = date(2026, 8, 20)


def test_the_pack_lists_only_articles_primarily_about_the_company():
    """On AVGO the pack carried 188 articles of which 127 were `mentioned` or
    `unrelated` — coverage the sentiment node had already judged not about
    the company — consuming 39% of the whole pack, while the debate cited
    news once in 25 claims."""
    pack = port.build_evidence_pack(_state(news_digest=_digest_with_mix()))

    assert "Acme beats estimates" in pack
    assert "Acme raises guidance" in pack
    assert "Sector roundup" not in pack
    assert "Unrelated company" not in pack


def test_the_pack_says_how_many_articles_it_withheld():
    """Silent omission is the failure the NOT RUN blocks exist to prevent: a
    debater shown two articles with no further comment reads that as the
    whole feed."""
    pack = port.build_evidence_pack(_state(news_digest=_digest_with_mix()))

    assert "2 further article(s)" in pack
    assert "not evidence of quiet news flow" in pack
    assert "2 of 9 vendor article(s) shown" in pack


def test_a_feed_with_nothing_about_the_company_says_so_loudly():
    """Distinct from 'news did not run' and from neutral news. Two articles
    about other companies must not read as two articles about this one."""
    digest = _digest_with_mix()
    digest.items = [i for i in digest.items if i.relevance != "primary"]

    pack = port.build_evidence_pack(_state(news_digest=digest))

    assert "NO article in the window was primarily about this company" in pack
    assert "ABSENCE of news evidence" in pack


def test_the_pack_filter_is_the_same_policy_the_sentiment_node_uses():
    """Keyed on AGGREGATED_RELEVANCE, not a literal, so the pack and the
    aggregate can never disagree about what counts as evidence about this
    company. Widening the policy must widen both at once."""
    from app.agent.trading.domain import news_digest as nd

    digest = _digest_with_mix()
    before = port.build_evidence_pack(_state(news_digest=digest))
    assert "Sector roundup" not in before

    port_mod = port
    original = port_mod.AGGREGATED_RELEVANCE
    try:
        port_mod.AGGREGATED_RELEVANCE = frozenset({"primary", "mentioned"})
        after = port.build_evidence_pack(_state(news_digest=digest))
    finally:
        port_mod.AGGREGATED_RELEVANCE = original

    assert "Sector roundup" in after
    assert "Unrelated company" not in after
    assert original is nd.AGGREGATED_RELEVANCE


def test_the_pack_carries_every_analyst_report_that_ran():
    state = _state(
        technical_report=TechnicalReport(
            ticker="ACN",
            as_of_date=date(2026, 8, 19),
            data_source="yfinance",
            bars_used=252,
            indicators=TechnicalIndicators(last_close=327.55, rsi_14=41.2033),
            interpretation="Momentum is soft.",
        ),
        news_digest=NewsDigest(
            ticker="ACN",
            as_of_date=date(2026, 8, 19),
            window_start=date(2026, 8, 5),
            items=[
                NewsItem(
                    headline="Acme beats estimates",
                    published_date=date(2026, 8, 18),
                    source="reuters",
                    url="https://example.com/1",
                    summary="Results above expectations.",
                    sentiment="positive",
                    relevance="primary",
                )
            ],
            raw_article_count=1,
            deduped_count=1,
            dropped_out_of_window=0,
            dropped_missing_date=0,
            truncated_by_cap=False,
        ),
        sentiment_summary=SentimentSummary(
            ticker="ACN",
            as_of_date=date(2026, 8, 19),
            positive=1,
            negative=0,
            neutral=0,
            net_score=1.0,
            article_count=1,
            excluded_by_relevance=0,
        ),
    )

    pack = port.build_evidence_pack(state)

    assert "operating margin of 34.1%" in pack
    assert "Momentum is soft." in pack
    assert "Acme beats estimates" in pack
    assert "net_score" in pack
    assert "NOT RUN" not in pack


def test_the_pack_layout_does_not_depend_on_which_legs_ran():
    """A pack whose section order changes with the run shape is a pack whose
    cache never hits."""
    full = port.build_evidence_pack(_state())
    partial = port.build_evidence_pack({"ticker": "ACN"})
    heads = lambda pack: [
        line.split(":")[0] for line in pack.splitlines() if line[:1].isupper() and ":" in line
    ][:5]
    assert heads(full)[:4] == heads(partial)[:4]


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def test_the_debate_model_follows_the_project_wide_setting():
    """One knob. TRADING_DEBATE_MODEL still overrides it for a one-off run,
    but the default is whatever LLM_CLAUDE_MODEL says, like every other
    node."""
    import os

    from app.agent.researcher import AGENT_MODEL

    if not os.getenv("TRADING_DEBATE_MODEL"):
        assert port.DEBATE_MODEL == AGENT_MODEL


def test_the_debate_model_is_priced_so_the_budget_assertion_can_fire():
    """An unpriced model makes every turn cost None, which sums to 0.00 and
    can never trip the ceiling — the budget assertion would be silently
    absent rather than merely loose."""
    from app.agent.researcher import _MODEL_PRICING

    assert port.DEBATE_MODEL in _MODEL_PRICING


def test_cost_is_computed_at_the_debate_model_s_rate_not_the_researcher_s():
    """log_cost hardcoded AGENT_MODEL for both the label and the price, which
    was correct by coincidence. It still has to take the model, because
    TRADING_DEBATE_MODEL can point the debate somewhere else — and a Sonnet
    debate priced as Haiku is understated ~3x, which is exactly the wrong
    number for a budget assertion to wave through."""
    from app.agent.researcher import _compute_cost, UsageSummary

    usage = UsageSummary()
    usage.input_tokens = 1_000_000
    usage.output_tokens = 0

    assert _compute_cost(usage, "claude-sonnet-5") == 3.0
    assert _compute_cost(usage, "claude-haiku-4-5-20251001") == 1.0


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-haiku-4-5-20251001", False),
        ("claude-sonnet-4-5-20250929", False),
        ("claude-opus-4-5", False),
        ("claude-sonnet-5", True),
        ("claude-opus-5", True),
        ("claude-sonnet-4-6", True),
        ("some-future-model", False),
    ],
)
def test_reasoning_params_are_only_sent_to_models_that_accept_them(model, expected):
    """Adaptive thinking and output_config.effort are REJECTED by 4.5-era
    models. An unknown id falls through to sending neither, because omitting
    them is valid everywhere and sending them is not — the safe default is
    the one that still runs."""
    assert port.supports_adaptive_thinking(model) is expected


@pytest.mark.anyio
async def test_a_45_era_model_is_not_sent_thinking_or_effort(monkeypatch):
    monkeypatch.setattr(port, "DEBATE_MODEL", "claude-haiku-4-5-20251001")
    client = _FakeClient([_payload()])

    await port.run_debate_turn(_state(), "bull", 0, client=client)

    call = client.messages.calls[0]
    assert "thinking" not in call
    assert "output_config" not in call
    assert call["tool_choice"]["disable_parallel_tool_use"] is True


@pytest.mark.anyio
async def test_a_46_era_model_is_sent_thinking_and_effort(monkeypatch):
    monkeypatch.setattr(port, "DEBATE_MODEL", "claude-sonnet-5")
    client = _FakeClient([_payload()])

    await port.run_debate_turn(_state(), "bull", 0, client=client)

    call = client.messages.calls[0]
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": port.DEBATE_EFFORT}


@pytest.mark.anyio
async def test_the_running_debate_cost_is_asserted_against_the_budget(monkeypatch):
    """Fires as soon as the total crosses the ceiling, not at the end: an
    assertion cannot refund a turn already paid for."""
    monkeypatch.setattr(port, "log_cost", lambda *a, **k: 0.30)
    turns = [_turn(0, "bull", ["a"])]
    turns[0].estimated_cost_usd = 0.30
    client = _FakeClient([_payload()])

    with pytest.raises(AssertionError, match="per-debate budget"):
        await port.run_debate_turn(
            _state(debate_turns=turns), "bear", 1, client=client
        )
