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


# The three below use HEX factor ids on purpose. `_content_id` (risk_port)
# mints `RF` + a 4-char SHA-1 prefix in uppercase hex, so a real slate looks
# like RF487E/RFC50B/RF6ECA — while every other fixture in this file (and in
# test_risk_ledger/test_risk_port) still says RF00, the shape ids had before
# the 2026-08-26 content-hash change. That gap is exactly how `RF\d+` in
# _REF_PATTERN survived: the suite only ever fed it the ~15% of the id space
# that happens to be all digits. Live NFLX memo, 2026-08-26: five factors
# cited, four hex, and one rendered.

def _hex_ledger():
    """A ledger whose ids are shaped the way `_content_id` really makes them."""
    return [_ledger()[0].model_copy(update={"factor_id": "RF487E"})]


def test_extract_refs_finds_a_hex_factor_id():
    refs = port.extract_refs("structural downtrend intact [RFC50B]")
    assert refs == ["RFC50B"]


def test_resolve_refs_flags_an_unresolved_hex_factor_id():
    ledger_by_id = {e.factor_id: e for e in _hex_ledger()}
    texts = ["cites a real one [RF487E] and an invented one [RFDEAD]"]
    assert port.resolve_refs(texts, {}, ledger_by_id) == ["RFDEAD"]


def test_render_evidence_includes_a_hex_id_factor():
    ledger_by_id = {e.factor_id: e for e in _hex_ledger()}
    lines = port._render_evidence(["the decisive risk is [RF487E]"], {}, ledger_by_id)
    assert len(lines) == 1
    assert lines[0].startswith("[RF487E]")


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
async def test_a_blocked_research_manager_call_still_logs_its_cost(monkeypatch):
    """log_cost must run before the fabrication guard's raise: the guard
    fires only after the LLM call already spent real tokens, and logging
    after the raise meant a blocked call's spend never reached
    cost-log.jsonl (trading-agent-known-gaps.md, FIG, 2026-08-26)."""
    logged = []
    monkeypatch.setattr(port, "log_cost", lambda *a, **k: logged.append(a[:2]) or 0.02)

    bait = _research_payload(thesis="Revenue could reach $9,999 million [C:margin-hold].")
    coro, _ = _run([bait])

    with pytest.raises(port.SynthesisFabricationError):
        await coro

    assert ("ACN", "trading-research-manager") in logged


@pytest.mark.anyio
async def test_a_blocked_risk_judge_call_still_logs_its_cost(monkeypatch):
    logged = []
    monkeypatch.setattr(port, "log_cost", lambda *a, **k: logged.append(a[:2]) or 0.02)

    bait = _risk_payload(reasoning="Losses could hit $12,345 thousand [RF00].")
    coro, _ = _run([_research_payload(), bait])

    with pytest.raises(port.SynthesisFabricationError):
        await coro

    assert ("ACN", "trading-research-manager") in logged  # the earlier, clean call
    assert ("ACN", "trading-risk-judge") in logged          # the blocked call itself


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
# Post-hoc memo verification (Phase 7) — verify_decision_memo re-checks the
# FULLY ASSEMBLED memo, independent of the per-call guards above that only
# ever see one call's own payload fields. Same containment methodology
# (_numeric_guard/_flag_debate_numbers), not a second implementation.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_a_clean_assembled_memo_passes_post_hoc_verification():
    coro, _ = _run([_research_payload(), _risk_payload()])
    memo = await coro

    result = port.verify_decision_memo(memo, _state(), _ledger(), _debate_turns())

    assert result.passed
    assert result.unbacked_numbers == []
    assert result.unresolved_references == []


@pytest.mark.anyio
async def test_a_number_spliced_into_the_assembled_memo_fails_verification():
    """The per-call guards already passed on the way to `memo` — this
    proves the post-hoc check catches a fabrication introduced AFTER
    generation (e.g. by assembly or rendering), not just re-deriving what
    the guard already knew."""
    coro, _ = _run([_research_payload(), _risk_payload()])
    memo = await coro
    corrupted = memo.model_copy(update={
        "reasoning": memo.reasoning + " Margin could reach 91.4 next year."
    })

    result = port.verify_decision_memo(corrupted, _state(), _ledger(), _debate_turns())

    assert not result.passed
    assert "91.4" in result.unbacked_numbers


# ---------------------------------------------------------------------------
# Corpus tiering (2026-08-27). `_numeric_corpus` used to merge the analyst
# reports with the debate's own claims and the risk ledger, so a figure the
# DEBATE invented was already "somewhere upstream" by the time the memo
# cited it -- which is the whole of what exact containment tests. The
# post-hoc check now runs twice, against grounded-only and against
# grounded+derived, and the difference is the finding.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_a_figure_only_the_debate_states_is_reported_as_debate_originated():
    """The number has an antecedent (the debate said it), so it is NOT
    unbacked -- but no analyst report contains it, so its only source is a
    model. That distinction is invisible against a merged corpus."""
    coro, _ = _run([_research_payload(), _risk_payload()])
    memo = await coro
    turns = _debate_turns()
    invented = turns[0].model_copy(deep=True)
    invented.payload.claims[0].text += " Financing could reach 88.6 billion."
    corrupted = memo.model_copy(update={
        "reasoning": memo.reasoning + " Financing could reach 88.6 billion."
    })

    result = port.verify_decision_memo(
        corrupted, _state(debate_turns=[invented, *turns[1:]]), _ledger(),
        [invented, *turns[1:]],
    )

    assert "88.6" in result.debate_originated_numbers
    assert "88.6" not in result.unbacked_numbers


@pytest.mark.anyio
async def test_debate_originated_numbers_do_not_gate():
    """Reported, never blocking. A memo that states any growth rate carries
    derived figures the reports do not contain verbatim; gating on those
    would fail nearly every real memo. Live check, three Phase 9 memos: the
    only debate-originated figures were AVGO's 78% (63,887/35,819-1) and
    5.4 (7,570-2,185), both sound derivations from grounded endpoints."""
    coro, _ = _run([_research_payload(), _risk_payload()])
    memo = await coro
    turns = _debate_turns()
    invented = turns[0].model_copy(deep=True)
    invented.payload.claims[0].text += " Growth of 88.6 percent."
    corrupted = memo.model_copy(update={
        "reasoning": memo.reasoning + " Growth of 88.6 percent."
    })

    result = port.verify_decision_memo(
        corrupted, _state(debate_turns=[invented, *turns[1:]]), _ledger(),
        [invented, *turns[1:]],
    )

    assert result.debate_originated_numbers
    assert result.passed


@pytest.mark.anyio
async def test_a_figure_the_analyst_report_states_is_neither_flag():
    coro, _ = _run([_research_payload(), _risk_payload()])
    memo = await coro
    # 34.1% is in the fundamentals summary -- see _state().
    corrupted = memo.model_copy(update={
        "reasoning": memo.reasoning + " Operating margin stands at 34.1%."
    })

    result = port.verify_decision_memo(corrupted, _state(), _ledger(), _debate_turns())

    assert "34.1" not in result.unbacked_numbers
    assert "34.1" not in result.debate_originated_numbers


@pytest.mark.anyio
async def test_an_unresolvable_reference_spliced_into_the_assembled_memo_fails_verification():
    coro, _ = _run([_research_payload(), _risk_payload()])
    memo = await coro
    corrupted = memo.model_copy(update={
        "reasoning": memo.reasoning + " A latent risk exists [RF99]."
    })

    result = port.verify_decision_memo(corrupted, _state(), _ledger(), _debate_turns())

    assert not result.passed
    assert result.unresolved_references == ["RF99"]


def test_verification_works_standalone_not_only_after_run_synthesis():
    """Constructs a DecisionMemo directly, never through run_synthesis —
    proves the check is a real, independent re-derivation over whatever
    memo it's handed, not something that only works because it's reusing a
    flag the generation path already computed."""
    from app.agent.trading.domain.decision_memo import DecisionMemo

    memo = DecisionMemo(
        ticker="ACN", bull_case="clean", bear_case="clean",
        research_thesis="clean", risk_debate_summary="clean",
        technical_signal="clean", reasoning="Fabricated growth of 63.2%.",
        watch_items=[], verdict=Verdict.HOLD, confidence=0.5,
        data_as_of_date=date(2026, 8, 19), data_gaps=[], assumptions=[], evidence=[],
    )

    result = port.verify_decision_memo(memo, _state(), _ledger(), _debate_turns())

    assert not result.passed
    assert "63.2" in result.unbacked_numbers or "63.2%" in result.unbacked_numbers


def test_memo_verification_error_is_not_caught_by_the_per_call_guard_exceptions():
    """MemoVerificationError signals a different, more serious failure
    class than SynthesisFabricationError/SynthesisReferenceError (an
    assembly-step bug vs. an ordinary bad call) — it must not be an
    instance of either, or a caller's `except SynthesisFabricationError`
    (e.g. synthesizer_node's per-trial drop-and-continue) would silently
    swallow it as if it were just another droppable trial failure."""
    assert not issubclass(port.MemoVerificationError, port.SynthesisFabricationError)
    assert not issubclass(port.MemoVerificationError, port.SynthesisReferenceError)

    err = port.MemoVerificationError("assembled memo failed verification")
    with pytest.raises(port.MemoVerificationError):
        try:
            raise err
        except (port.SynthesisFabricationError, port.SynthesisReferenceError):
            pytest.fail("MemoVerificationError was caught by the per-call guard except clause")


def test_an_unbacked_number_confined_to_watch_items_passes_verification():
    """Live regression (ASML, 2026-08-26): watch_items is a GAP-ONLY field
    at generation time (run_risk_judge flags an unbacked watch_items number
    into data_gaps rather than blocking on it — same test file, watch_items
    test above), so the assembled memo is legitimately allowed to carry one.
    The first version of verify_decision_memo scanned watch_items as if it
    were load-bearing and raised MemoVerificationError on exactly this case
    — a stricter standard than generation time itself applies, which is the
    assembly-step bug this function exists to catch, not a fabrication."""
    from app.agent.trading.domain.decision_memo import DecisionMemo

    memo = DecisionMemo(
        ticker="ACN", bull_case="clean", bear_case="clean",
        research_thesis="clean", risk_debate_summary="clean",
        technical_signal="clean", reasoning="clean",
        watch_items=["Volume surge above 0.80x of 20-day average would change this."],
        verdict=Verdict.HOLD, confidence=0.5, data_as_of_date=date(2026, 8, 19),
        data_gaps=["1 number(s) in the Risk Judge's watch items did not appear "
                   "in any source and may be fabricated: 0.80"],
        assumptions=[], evidence=[],
    )

    result = port.verify_decision_memo(memo, _state(), _ledger(), _debate_turns())

    assert result.passed


def test_an_unbacked_number_confined_to_bull_or_bear_case_passes_verification():
    """Same principle as the watch_items case above, for the Research
    Manager's own gap-only fields (bull_case/bear_case) — see
    test_an_unbacked_number_in_bull_case_is_a_gap_not_a_block."""
    from app.agent.trading.domain.decision_memo import DecisionMemo

    memo = DecisionMemo(
        ticker="ACN", bull_case="Upside could reach $7,777 million.",
        bear_case="clean", research_thesis="clean", risk_debate_summary="clean",
        technical_signal="clean", reasoning="clean", watch_items=[],
        verdict=Verdict.HOLD, confidence=0.5, data_as_of_date=date(2026, 8, 19),
        data_gaps=["number 7777 did not appear in any source and may be fabricated"],
        assumptions=[], evidence=[],
    )

    result = port.verify_decision_memo(memo, _state(), _ledger(), _debate_turns())

    assert result.passed


def test_verification_does_not_scan_data_gaps_or_evidence():
    """data_gaps deliberately quotes already-flagged unbacked numbers (that
    IS why they're gaps); evidence is Python-rendered from resolved
    references, not model prose. Scanning either would just re-report
    numbers the pipeline already knows about, not find something new."""
    from app.agent.trading.domain.decision_memo import DecisionMemo

    memo = DecisionMemo(
        ticker="ACN", bull_case="clean", bear_case="clean",
        research_thesis="clean", risk_debate_summary="clean",
        technical_signal="clean", reasoning="clean",
        watch_items=[], verdict=Verdict.HOLD, confidence=0.5,
        data_as_of_date=date(2026, 8, 19),
        data_gaps=["number 77.7 did not appear in any source and may be fabricated"],
        assumptions=[], evidence=["[C:margin-hold] cites 77.7 in the debate"],
    )

    result = port.verify_decision_memo(memo, _state(), _ledger(), _debate_turns())

    assert result.passed


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


@pytest.mark.anyio
async def test_verify_or_raise_returns_the_verification_for_the_caller():
    """It used to return None. `synthesizer_node` now reads
    `debate_originated_numbers` off the result to append a data gap, so a
    silent None here would be an AttributeError on every clean run -- the
    kind of break that only shows up in production because the passing path
    is the one nobody asserts on."""
    from app.agent.trading.application import nodes

    coro, _ = _run([_research_payload(), _risk_payload()])
    memo = await coro

    result = nodes._verify_or_raise(memo, _state(), _ledger(), _debate_turns())

    assert result is not None
    assert result.passed
    assert isinstance(result.debate_originated_numbers, list)
