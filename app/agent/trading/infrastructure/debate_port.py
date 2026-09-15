"""The LLM side of the bull/bear debate: evidence pack, one forced tool call,
the guardrails, and the vault transcript.

Structured output follows the direct-SDK pattern the Phase 3/4 ports use — a
single tool with a forced `tool_choice`, validated by pydantic — rather than
free text that has to be parsed back. The model produces argument content and
nothing else; every index, counter and side label is assigned in Python.

On guardrails: the consistent finding in the multi-agent-debate literature
(Du et al. 2023; Liang et al. 2023 on Degeneration-of-Thought; the sycophancy
work) is that debate reliably produces CONVERGENCE, and convergence is not
evidence of correctness. Two instances of one base model drift toward
agreement because agreement is what the pretraining distribution rewards.
Directions recalled rather than re-read — verify before citing any of it.
Four of the five counters here are enforced by pydantic or Python, because a
prompt-only guardrail is the kind that degrades silently.

Debate has the opposite failure mode too, and this module found it: total
entrenchment. Across all 42 vault transcripts, 249 turns are `hold` or
`sharpen` and NOT ONE is `concede`, while debaters concede in the argument
prose. `check_concession`'s docstring has the diagnosis. Half the fix there is
structural (a partial concession is now expressible and still validated); the
other half — getting the model to route a prose concession into
`concession_trigger` — is prompt-only and therefore exactly the kind of
guardrail this module distrusts. It is written that way ON PURPOSE: the
alternative is scanning the argument text for "I concede", and that does not
work. Of 21 occurrences of the word across the vault, 18 are a debater saying
the OPPONENT concedes something (an attack, the opposite of a concession) and
one is a negation ("None of this is a reason to concede..."). A keyword
counter would report ~21 concessions where there are 2 and invert the
direction of 18 of them. So the prompt asks, and the reporting side states
plainly what a zero means rather than claiming nobody moved.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

# create_with_temperature_fallback lives with the clients now, so the query
# decomposer can use it too; re-exported for risk_port and synthesis_port.
from app.agent.trading.infrastructure.structured_call import (
    assert_within_budget,
    call_with_schema_retry,
    force_crash,
)
# create_with_temperature_fallback is re-exported on purpose: risk_port and
# the determinism scripts import it from here, and several tests patch it
# at this name. It reads as unused to a naive import scan; it is not.
from app.infrastructure.llm import LLMClient, create_with_temperature_fallback, get_client
from app.infrastructure.llm.models import model_for, warn_if_unpriced

from app.agent.researcher import (
    UsageSummary,
    _save_output,
    log_cost,
)
from app.agent.trading.application.nodes import ANALYST_OUTPUTS
from app.agent.trading.domain.debate import (
    DebateTurn,
    DebateTurnPayload,
    Side,
    canonical_claims,
)
from app.agent.trading.domain.news_digest import AGGREGATED_RELEVANCE
from app.agent.trading.domain.sanitize import EXTERNAL_TEXT_FRAMING
from app.agent.trading.infrastructure.cost_log import new_event_id, record_cost_event
from app.agent.trading.infrastructure.technical_interpreter_port import (
    _PERIOD_LABEL,
    _flag_unmatched_numbers_against,
    derive_relations,
)

# The project-wide model from .env (LLM_CLAUDE_MODEL), same as every other
# node. TRADING_DEBATE_MODEL still overrides it for a one-off run without
# moving the whole pipeline.
#
# Worth knowing what this trades away: the debate is the one phase where
# reasoning quality IS the deliverable. A bull that cannot construct a real
# counterargument produces a transcript that looks like a debate and isn't,
# and that failure is invisible to both exit criteria — they test
# termination and resume, not argument quality. Read a transcript by hand
# after changing this, because no assertion here will tell you.
DEBATE_MODEL = model_for("debate")

# Room for adaptive thinking plus the tool call. Thinking tokens count
# against this, so the 1200 that fit a text-only turn does not fit here.
DEBATE_MAX_TOKENS = 4000

# Whole-debate ceiling, not per turn. Deliberately model-independent: it is a
# runaway and prompt-bloat trip wire, not a target, so it stays put when
# LLM_CLAUDE_MODEL moves.
#
# MEASURED, six turns over the technical report alone (AVGO, 2026-08-23):
#   claude-sonnet-5   $0.1506 total, ~$0.025/turn, 80s wall clock
#   claude-haiku-4-5  ~$0.005/turn, so ~$0.03 for the same six turns
# Input grows ~1.8k/turn as the transcript does. A full four-report pack will
# be dearer than either figure; re-measure before treating 0.35 as a margin
# rather than a ceiling.
DEBATE_BUDGET_USD = 0.35

# Thinking ON, effort LOW — where the model supports it. This started as
# {"type": "disabled"} on the reasoning that one forced tool call with a
# <=200 word argument has nothing for a thinking budget to buy. Two live
# turns disproved it: BOTH first attempts came back with `stance` missing and
# a serialized "...</submit_argument>" string stuffed inside `argument` — the
# model had half-written the tool call as text. That is the documented
# thinking-disabled failure mode, and it cost a retry on 2 of 2 turns.
DEBATE_THINKING: dict[str, Any] = {"type": "adaptive"}
DEBATE_EFFORT = "low"

# Adaptive thinking and output_config.effort exist on the 4.6-and-later
# families and are REJECTED by 4.5-era models — Haiku 4.5 and Sonnet 4.5 take
# the older {"type": "enabled", "budget_tokens": N} form and error on
# `effort` outright. Now that DEBATE_MODEL follows LLM_CLAUDE_MODEL, which
# points at Haiku 4.5 today, sending them unconditionally would 400 every
# turn.
#
# Prefixes rather than a version comparison, because the id format is not a
# reliable ordering ("claude-sonnet-5" sorts below "claude-sonnet-4-6"), and
# an UNKNOWN id falls through to sending NEITHER: omitting both is valid on
# every model while sending them is not, so the safe default is the one that
# still runs. A model added here without checking gets a 400 on the first
# turn, which is loud and cheap.
_ADAPTIVE_THINKING_MODELS = (
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)


def supports_adaptive_thinking(model: str) -> bool:
    return model.startswith(_ADAPTIVE_THINKING_MODELS)


def reasoning_config(model: str, temperature: float | None) -> dict:
    """The `thinking`/`output_config`/`temperature` kwargs for one call.

    Production (`temperature=None`, the default everywhere this is called):
    unchanged behavior — adaptive thinking on for the models that support it,
    no explicit `temperature` sent at all.

    An EXPLICIT temperature (Phase 6's determinism/stability checks, which
    need `temperature=0` and a fixed low temperature respectively) disables
    thinking outright, on every model, regardless of the value requested.
    Extended/adaptive thinking requires the API's default temperature and
    rejects an explicit one alongside it — sending both is a 400, and a
    reproducibility check that intermittently 400s on the very call it is
    trying to make deterministic is worse than no check. Determinism claims
    about the *thinking-enabled* production path are therefore a
    correlational claim of "this held with thinking off", not a proof that
    holds with it on — recorded in the finding, not hidden by it.
    """
    if temperature is not None:
        return {"temperature": temperature}
    if supports_adaptive_thinking(model):
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}
    return {}


warn_if_unpriced(DEBATE_MODEL, "debate", DEBATE_BUDGET_USD)

# Forced-failure hooks for the resume tests. Deliberately in the port rather
# than the node: variant B has to die AFTER the API call and before the node
# returns, which is the window a `kill -9` would land in and the one the
# add-reducer's re-execution behaviour is actually tested by.
_CRASH_AT = os.getenv("DEBATE_CRASH_AT_TURN")
_CRASH_WHEN = os.getenv("DEBATE_CRASH_WHEN", "before")   # "before" | "after"


def _maybe_crash(turn_index: int, when: str) -> None:
    if _CRASH_AT is None or int(_CRASH_AT) != turn_index or _CRASH_WHEN != when:
        return
    force_crash("debate", f"{when} turn {turn_index} (DEBATE_CRASH_AT_TURN)")


# ---------------------------------------------------------------------------
# Prompts — symmetric by construction
# ---------------------------------------------------------------------------
#
# Both sides are built from one template with a single stance slot, so
# `BULL_SYSTEM.replace(BULL_STANCE, "") == BEAR_SYSTEM.replace(BEAR_STANCE,
# "")` holds structurally rather than by anyone remembering to mirror an
# edit. Asserted as a unit test: any asymmetry becomes a permanent confound
# in every transcript reasoned over later, and it goes invisible after a few
# prompt edits.

BULL_STANCE = """\
You argue the BULL case: that the evidence supports owning this security.
Your opponent argues the bear case."""

BEAR_STANCE = """\
You argue the BEAR case: that the evidence argues against owning this security.
Your opponent argues the bull case."""

_SYSTEM_TEMPLATE = """\
You are one side of a structured, adversarial equity research debate.

{STANCE}

You will be given an EVIDENCE PACK containing the analyst reports produced for
this ticker, and the transcript of the debate so far. Argue from the pack.

{EXTERNAL_TEXT_FRAMING}

HARD RULES — these are checked in code after you answer:

1. EVERY figure you write must appear VERBATIM in the evidence pack. Do not
   compute, re-derive, annualize, or restate a number in a different unit.
   Your job is to cite the analysts' numbers, not to produce new ones. A
   figure you derived is, by construction, unbacked.
1a. Where the pack gives a "Computed relations" block, those comparisons are
   worked out in code and are AUTHORITATIVE. State them as given. Do not
   work out a comparison yourself from the raw indicator values, and never
   contradict the block — including the bands it names, such as whether RSI
   is overbought or oversold. A relation line is also the cleanest thing to
   quote for a claim about two values, because it carries both.
2. Every claim carries an `evidence_ref`. If the claim rests on a report,
   name that report and quote it: `evidence_quote` must be a VERBATIM,
   CONTIGUOUS span of at most 25 words copied out of that report — never two
   fragments joined with "...". If the claim is reasoning over other claims
   rather than a report-backed fact, set evidence_ref='none' and
   evidence_quote='none'.
   NEVER send an empty string for any field. Where a field does not apply,
   send the literal string 'none'.
3. `claim_id` is a short stable slug. When you restate a claim you or your
   opponent already made, REUSE THE EXISTING id. Invent a new id only for a
   genuinely new assertion.
4. `rebuts` lists the opponent claim_ids you are directly attacking. It may
   be empty only on the opening turn.
5. Stance — your OVERALL posture this turn:
   - 'hold'    — you are maintaining your position against the rebuttal.
   - 'sharpen' — you are narrowing or qualifying your own earlier claim.
   - 'concede' — the opposing claim has overturned your position. Rare.
6. `concession_trigger` is a SEPARATE axis from stance. It names the ONE
   opponent claim_id you are accepting as correct this turn, and it applies
   on ANY stance. Accepting a point you cannot answer while holding your
   overall case is the normal, expected shape: stance='hold' with
   `concession_trigger` naming that claim. Reserve stance='concede' for the
   rarer case where accepting it overturns your side.
   IF YOUR ARGUMENT TEXT ACCEPTS AN OPPOSING POINT — "I concede X", "that is
   a genuine overhang", "the bear is right about X", "X is real but" — YOU
   MUST NAME THAT CLAIM'S id IN `concession_trigger`. A concession written
   only in prose is invisible to everything downstream, which then reports
   that neither side moved. The id must be an OPPONENT claim_id already in
   the transcript; your own earlier claim is not a concession, and neither
   is an id nobody made. If you accepted nothing this turn, send 'none'.

A report marked "NOT RUN" is missing evidence, not neutral evidence. Do not
infer anything from its absence, and do not argue from it in either direction.

Do not concede to be agreeable, and do not manufacture disagreement. If the
evidence genuinely does not support your side on a point, name it in
`concession_trigger` and argue the points where it does.

Call `submit_argument` exactly once. Say nothing else."""

BULL_SYSTEM = _SYSTEM_TEMPLATE.replace("{STANCE}", BULL_STANCE).replace(
    "{EXTERNAL_TEXT_FRAMING}", EXTERNAL_TEXT_FRAMING
)
BEAR_SYSTEM = _SYSTEM_TEMPLATE.replace("{STANCE}", BEAR_STANCE).replace(
    "{EXTERNAL_TEXT_FRAMING}", EXTERNAL_TEXT_FRAMING
)

_STANCE_BY_SIDE = {"bull": BULL_STANCE, "bear": BEAR_STANCE}
_SYSTEM_BY_SIDE = {"bull": BULL_SYSTEM, "bear": BEAR_SYSTEM}


# ---------------------------------------------------------------------------
# Evidence pack
# ---------------------------------------------------------------------------

def _not_run(name: str) -> str:
    """A missing analyst leg is stated, not omitted.

    `--only news` leaves fundamentals_report and technical_report as None. If
    the pack simply had no fundamentals section, a debater would read the
    silence as neutrality — the exact error the news caveats exist to prevent
    one layer up.
    """
    return (
        f"{name.upper()}: NOT RUN — no {name} evidence is available for this "
        f"debate. Do not infer anything about {name} from its absence."
    )


def _render_technical(report, *, quotable: bool = False) -> str:
    """Relations FIRST, then the raw values, then the prose.

    The relations block is Phase 3's `derive_relations` — the comparisons
    computed in Python precisely because a model asked to work them out from
    raw numbers gets them wrong. The pack used to hand the debaters the JSON
    and nothing else, throwing that away, and on the first live Haiku turns
    BOTH sides called an RSI of 38.7 "oversold". It is not, and the relations
    block says so in as many words. Every number in those turns was real, so
    the numeric guard had nothing to catch — the same shape as the MSFT
    moving-average error that made `derive_relations` exist.

    It also gives the debaters something QUOTABLE. `evidence_quote` is a
    single contiguous span, and a trend claim rests on two values that sit
    far apart in the JSON, so an honest citation of both was a splice and got
    flagged. One relation line carries both values and the comparison
    between them.

    The JSON stays — in the pack the number-fabrication guard scans. It is
    the only source of full precision, and a claim in prose that turns on the
    fourth decimal has nowhere else to cite.

    `quotable=True` drops the JSON line. Containment on a serialized dict
    lets `evidence_quote` cite `macd_histogram":0.3556307403914323` verbatim
    — the check passes, because it IS in the pack, but the debater is
    grepping the raw blob rather than citing anything the analyst actually
    said. Found live (ACN, technical-only pack, 2026-08-24): 2 of 4 claims in
    one turn quoted a raw JSON key:value fragment this way. Used only for the
    quote-check corpus (`quotable_texts`) — `build_evidence_pack` still gets
    the full render, so the number-fabrication guard keeps the precision
    backstop and a faithfully-copied figure in argument prose is not falsely
    flagged as fabricated.
    """
    relations = "\n".join(f"- {r}" for r in derive_relations(report.indicators))
    header = (
        f"TECHNICAL (as of {report.as_of_date}, {report.data_source}, "
        f"{report.bars_used} bars):\n"
        f"Computed relations (AUTHORITATIVE — worked out in code, not by a "
        f"model. State them as given; never contradict them, and never "
        f"re-derive a comparison yourself from the raw values below):\n"
        f"{relations}\n"
    )
    if not quotable:
        header += f"Indicators (full precision): {report.indicators.model_dump_json()}\n"
    return header + f"Interpretation: {report.interpretation}"


def _render_news(digest) -> str:
    """Only the articles the sentiment aggregate counts, and a line saying so.

    The pack used to carry every item the vendor returned. On AVGO that was
    188 articles of which 127 were `mentioned` or `unrelated` — coverage the
    sentiment node had ALREADY judged not primarily about the company — and
    they consumed 39% of the whole evidence pack. The debate cited news once
    in 25 claims, so that was context nobody read, paid for on every turn.

    Filtered on AGGREGATED_RELEVANCE rather than a literal, so the pack and
    the sentiment aggregate cannot disagree about what counts as evidence
    about this company. One constant, one policy.

    The omission is STATED, not silent. A debater shown 61 articles with no
    further comment reads that as the whole feed, and the pack's own rule —
    the same one behind the NOT RUN blocks — is that absence must be visible.
    """
    shown = [i for i in digest.items if i.relevance in AGGREGATED_RELEVANCE]
    hidden = len(digest.items) - len(shown)

    header = (
        f"NEWS ({digest.window_start} to {digest.as_of_date}, "
        f"{len(shown)} of {digest.raw_article_count} vendor article(s) shown, "
        f"truncated_by_cap={digest.truncated_by_cap}):"
    )
    notes = []
    if hidden:
        notes.append(
            f"{hidden} further article(s) in the feed mentioned the company or "
            f"were unrelated to it and are NOT listed. Their absence is a "
            f"filtering decision, not evidence of quiet news flow."
        )
    if not shown:
        notes.append(
            "NO article in the window was primarily about this company. That is "
            "an ABSENCE of news evidence, not neutral news — do not argue from "
            "it in either direction."
        )

    lines = [header, *notes]
    for item in shown:
        lines.append(
            f"- [{item.published_date}] ({item.relevance}/{item.sentiment}) "
            f"{item.headline}: {item.summary}"
        )
    return "\n".join(lines)


def _render_sentiment(summary) -> str:
    return (
        f"SENTIMENT (as of {summary.as_of_date}): net_score {summary.net_score:+.3f} "
        f"over {summary.article_count} article(s) primarily about the company "
        f"(+{summary.positive} / -{summary.negative} / ={summary.neutral}); "
        f"{summary.excluded_by_relevance} excluded as not primarily about it. "
        f"An article_count of 0 is an absence of evidence, not neutral evidence."
    )


def report_texts(state) -> dict[str, str]:
    """One text block per `evidence_ref` value, keyed by that value.

    Also the corpus the quote check runs against, which is why it is built
    once and reused rather than re-rendered per claim.
    """
    fundamentals = state.get("fundamentals_report")
    technical = state.get("technical_report")
    digest = state.get("news_digest")
    sentiment = state.get("sentiment_summary")
    return {
        "fundamentals": (
            f"FUNDAMENTALS:\n{fundamentals.summary}"
            if fundamentals is not None
            else _not_run("fundamentals")
        ),
        "technical": (
            _render_technical(technical)
            if technical is not None
            else _not_run("technical")
        ),
        "news": _render_news(digest) if digest is not None else _not_run("news"),
        "sentiment": (
            _render_sentiment(sentiment)
            if sentiment is not None
            else _not_run("sentiment")
        ),
    }


def quotable_texts(state) -> dict[str, str]:
    """`report_texts`, but for the corpus `check_quotes` validates against.

    Identical for every source except technical, where the raw indicators
    JSON is dropped — see `_render_technical`'s `quotable` docstring for why.
    `build_evidence_pack` keeps calling `report_texts` (unchanged), so the
    number-fabrication guard still has the JSON as ground truth; only the
    quote check loses it.
    """
    texts = report_texts(state)
    technical = state.get("technical_report")
    if technical is not None:
        texts["technical"] = _render_technical(technical, quotable=True)
    return texts


def build_evidence_pack(state) -> str:
    """Built off ANALYST_OUTPUTS order so a partial run produces the same
    section order as a full one — a pack whose layout changes with the run
    shape is a pack whose cache never hits."""
    texts = report_texts(state)
    order = list(ANALYST_OUTPUTS) + ["sentiment"]
    return (
        f"EVIDENCE PACK — {state['ticker'].upper()}\n\n"
        + "\n\n".join(texts[name] for name in order)
    )


def render_transcript(turns: list[DebateTurn]) -> str:
    """The debate so far, as the next speaker sees it."""
    if not turns:
        return "TRANSCRIPT: empty — this is the opening turn."
    lines = ["TRANSCRIPT SO FAR:"]
    for turn in turns:
        lines.append(
            f"\n[turn {turn.turn_index} · round {turn.round_num} · "
            f"{turn.side.upper()} · stance={turn.payload.stance}"
            + (
                f" · concedes to {turn.payload.concession_trigger}"
                + ("" if turn.payload.stance == "concede" else " (partial)")
                if turn.payload.concession_trigger
                else ""
            )
            + "]"
        )
        lines.append(turn.payload.argument)
        for claim in turn.payload.claims:
            quote = f' "{claim.evidence_quote}"' if claim.evidence_quote else ""
            lines.append(
                f"  · {claim.claim_id} [{claim.evidence_ref}]: {claim.text}{quote}"
            )
        if turn.payload.rebuts:
            lines.append(f"  rebuts: {', '.join(turn.payload.rebuts)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

# JSON Schema keywords a strict tool schema rejects. Dropped from the wire
# format only — pydantic still enforces every one of them on the way back in,
# so the constraint is not lost, just not advertised. Where the bound matters
# to the model (claims 1..5) it is restated in prose in the field
# description.
#
# `minimum`/`maximum` joined this set in Phase 6, found live: RiskScore's
# `severity`/`likelihood` (pydantic `ge=1, le=5`) 400'd every risk-panel turn
# with "For 'integer' type, properties maximum, minimum are not supported" —
# unlike Phase 5's array-length bounds, which were anticipated from the API
# docs, this one was not caught until a real call hit it. Same fix: state the
# 1-5 range in the field description (domain/risk.py), and rely on pydantic
# to still enforce it once the value comes back.
_STRICT_UNSUPPORTED = frozenset(
    {"minItems", "maxItems", "minLength", "maxLength", "pattern", "format", "minimum", "maximum"}
)


def _inline_refs(schema: dict) -> dict:
    """Splice $defs into the tree and drop the key.

    `model_json_schema()` emits `$defs` + `$ref` for the nested DebateClaim.
    $ref resolution inside a tool `input_schema` has not been reliable in my
    experience and cannot be verified from here, so the refs are inlined
    before sending. DebateClaim is flat by design, which keeps this walk to a
    single level and non-recursive.
    """
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                merged = walk(defs[ref.rsplit("/", 1)[-1]])
                # sibling keys (description, default) win over the target's
                merged.update({k: walk(v) for k, v in node.items() if k != "$ref"})
                return merged
            out = {
                k: walk(v)
                for k, v in node.items()
                if k not in _STRICT_UNSUPPORTED
            }
            # `strict: true` requires additionalProperties: false on every
            # object in the tree, and every property listed in `required` —
            # pydantic omits the defaulted ones, so they are added back here
            # rather than by deleting the defaults from the domain type.
            if out.get("type") == "object" and "properties" in out:
                out["additionalProperties"] = False
                out["required"] = list(out["properties"])
                # A `default` on a field the model is now REQUIRED to emit is
                # a contradiction, and the one it reads as permission to send
                # an empty string — which is the failure the 'none' sentinel
                # exists to avoid. Strip it from the wire schema; pydantic
                # keeps it for Python-side construction.
                for prop in out["properties"].values():
                    if isinstance(prop, dict):
                        prop.pop("default", None)
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


SUBMIT_TOOL = {
    "name": "submit_argument",
    "description": "Submit this turn's argument. Call exactly once.",
    # strict, so tool_use.input is guaranteed to match the schema. Without it
    # the first attempt at a turn came back flattened — the DebateClaim
    # fields hoisted to the top level beside `argument`, `stance` missing
    # entirely — on 3 of 3 live turns, costing a retry every time. A retry
    # loop is the one runaway the round cap cannot see, so removing the
    # reason to retry is worth more than handling the retry well.
    "strict": True,
    "input_schema": _inline_refs(DebateTurnPayload.model_json_schema()),
}


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

# Comma-aware, and a '-' counts as a sign only where it cannot be something
# else. Two lookbehinds, each patched from a real false positive rather than
# designed up front:
#
#   1. A RANGE separator, the Phase 3 lesson: reading every hyphen as a minus
#      turned a faithful "318.73-352.11" band into a fabricated "-352.11".
#   2. A HYPHENATED COMPOUND, found on the first live debate (AVGO,
#      2026-08-23): "the low-30s oversold zone" and "sub-50-SMA price" were
#      read as -30 and -50 and reported as fabricated figures. Two of the six
#      flags that run, so it is not a rare shape.
#
# The second lookbehind has to cover the digit as well as the sign. Blocking
# only "<letter>-<digits>" would leave the scanner free to start one
# character later and flag a bare "30" out of "low-30s" — the same false
# positive with the sign filed off.
_DEBATE_NUMBER = re.compile(
    r"(?<![\w.%,])(?<![A-Za-z]-)(-?\d[\d,]*\.?\d*)(%?)"
)


def _flag_debate_numbers(text: str, evidence_pack: str) -> list[str]:
    """Every figure in a debate turn must appear verbatim in the evidence pack.

    Containment rather than the Phase 3 tolerance match, and the difference
    matters. That guard works over ~10 well-separated TechnicalIndicators
    values at tolerance max(0.5, |kv|*0.02), where a fabricated number
    usually misses all of them. Scrape every number out of a fundamentals
    memo and there are a hundred-plus known values carrying the same bands;
    in dense regions those bands overlap and cover most of the number line, a
    fabricated figure lands inside somebody's band, and the guard returns []
    forever while reading as clean.

    Containment is also the semantically correct check here: the debater's
    job is to CITE the analysts' numbers, not compute new ones, so a figure
    that is not in the pack is unbacked whatever its value.

    Two faithful restatements are cleared before anything is flagged, both
    forced by live output rather than designed up front:

    1. ROUNDING, which is what prose does to figures. The first two live
       turns wrote "RSI of 41.2" for 41.2033 and "the 50-day at 330.12" for
       330.1245 — one false positive each, a 100% rate on day one, which
       would have taught the reader to skip the guard before it ever caught
       anything. Cleared by PRECISION, not tolerance: a figure clears only
       if some pack value rounds to it AT THE FIGURE'S OWN number of decimal
       places. "41.2" clears against 41.2033; "71.4" clears against nothing.
       That is far tighter than a +/-2% band and does not widen as the pack
       grows.
    2. PERCENT forms, where a faithful restatement legitimately differs from
       the source ("53%" for a volume_vs_20d_avg of 0.529, "22% above" for
       1.2153). Those are handed to the Phase 3 transforms, and only to them
       — the bare-number tolerance stage is deliberately not given veto
       power here, for the density reason above.

    3. SIGN, added 2026-08-29. The news feed says Viking "Cuts Share Stake In
       Microsoft Corp By 36.8%" and the debater wrote "Viking -36.8%" — the
       same figure, the sign carrying the direction the source states in
       words. Comparison is therefore on MAGNITUDE: "-36.8" clears against a
       pack "36.8" and vice versa.

       What that gives up, stated rather than discovered later: this guard no
       longer reports a sign INVERSION of a sourced figure — "-5.2%" written
       against a pack "+5.2%" now clears. That is the correct trade for THIS
       guard, whose question is whether a figure has a source at all, not
       whether it was read in the right direction; a magnitude present in the
       pack was not invented. Direction errors need a check that knows what
       the number means, which containment never did.

    Flags, never blocks. This guard is new enough to have unknown
    false-positive classes, and debate prose is looser than the templated
    technical interpretation, so expect more of them here than in Phase 3.
    """
    text = text.replace("−", "-")
    pack = evidence_pack.replace("−", "-").replace(",", "")

    pack_tokens = {m.group(1) for m in _DEBATE_NUMBER.finditer(pack)}
    # Magnitudes as well as signed tokens — see the SIGN note in the
    # docstring. Built from the same scan so the two can never disagree.
    pack_magnitudes = {t.lstrip("-") for t in pack_tokens}
    known: list[float] = []
    for token in pack_tokens:
        try:
            known.append(float(token))
        except ValueError:
            continue
    known_magnitudes = [abs(k) for k in known]

    # Second opinion, consulted ONLY for percent forms (see docstring).
    tolerance_flags = set(_flag_unmatched_numbers_against(text, known))

    scanned = _PERIOD_LABEL.sub("", text)
    flagged: list[str] = []
    for match in _DEBATE_NUMBER.finditer(scanned):
        raw, percent = match.group(1), match.group(2)
        bare = raw.replace(",", "")
        if bare in pack_tokens or bare.lstrip("-") in pack_magnitudes:
            continue
        if _is_rounding_of(bare.lstrip("-"), known_magnitudes):
            continue
        if percent:
            if not ({f"{raw}%", f"{raw}% above/below"} & tolerance_flags):
                continue   # a faithful transform of a pack value
            flagged.append(f"{raw}%")
        else:
            flagged.append(raw)

    # Order-preserving dedup: the same fabricated figure repeated four times
    # is one finding, not four.
    return list(dict.fromkeys(flagged))


# Unit conversion is a power of 1000: a filing reports "$63,887" million or
# "$14,462,836k", and prose says "$63.9B". Nothing else about the figure
# changes.
_SCALE_FACTORS = (1e3, 1e6, 1e9)


def _is_rounding_of(raw: str, known: list[float]) -> bool:
    """True when some pack value rounds to `raw` at `raw`'s own precision,
    at the same scale or a power-of-1000 away from it.

    Precision-scoped on purpose. A fixed tolerance widens the guard's blind
    spot as the pack grows; this one does not — "41.2" only ever clears
    against a value in [41.15, 41.25), whatever else is in the pack.

    SCALE was added 2026-08-27 after the Phase 9 battery measured this
    guard's precision on three live memos and found it inverted: every
    "may be fabricated" figure it reported was correct and every one was a
    millions-to-billions restatement it could not see — AVGO's 63.9
    ($63,887M revenue), 35.8 ($35,819M), 5.7 ($5,747M SBC), NFLX's 10.1
    ($10,149M OCF), ACN's 69.7 ($69,673M revenue). Meanwhile the one real
    fabrication in that battery went unreported. A guard whose warnings are
    reliably wrong is worse than no guard: it teaches the reader to skip
    the category, and then the true positive arrives in a list nobody reads.

    Scale clearing is deliberately restricted to figures carrying at least
    one DECIMAL PLACE, and that restriction is the whole safety argument.
    Dividing the pack by 1000 multiplies the number of values the guard will
    clear against, which is exactly the density problem `_flag_debate_numbers`
    exists to avoid — but only for coarse figures. A bare integer like "70"
    would clear against any pack value in [69500, 70500), a bucket 1000 wide,
    and "70" is precisely the shape of the fabricated figure this battery
    caught. A converted figure keeps its significant digits ("$63.9B", never
    "$64B") because keeping them is the point of converting; so requiring a
    decimal admits the real restatements and admits none of the round
    inventions. "63.9" still only ever clears against [63850, 63950).

    RESIDUAL, measured and accepted rather than engineered away: a
    one-decimal figure clears against a window 100 units wide at the
    1000-scale, so a coincidental match is possible in a dense corpus. Seen
    once, in the same Phase 9 re-audit: AVGO's "$2.2B" (pre-VMware SBC,
    legitimately 6.1% x $35,819M = $2,185M) cleared against an unrelated
    2,171 elsewhere in the corpus. The figure was sound and the clearance
    was luck. Tightening this by requiring an explicit magnitude unit does
    NOT help — the corpus reports in millions, so 2,171M reads as $2.17B and
    matches at the stated scale too. Containment on a rounded figure against
    a dense corpus is coarse by construction; the honest trade is 7 measured
    false positives removed against a coincidence rate that is bounded and
    documented. Precision here is what `debate_originated_numbers`
    (synthesis_port) exists to add, by asking a different question — does
    this figure have an analyst source at all — rather than a looser one.
    """
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return False
    fraction = raw.split(".")
    places = len(fraction[1]) if len(fraction) == 2 else 0
    if any(round(kv, places) == value for kv in known):
        return True
    if places == 0:
        return False   # see the docstring: no scale clearing for bare integers
    return any(
        round(kv / factor, places) == value
        for kv in known
        for factor in _SCALE_FACTORS
    )


# ---------------------------------------------------------------------------
# Direction guard — the numbers are right and the sentence about them is not
# ---------------------------------------------------------------------------

# Only words whose direction on the CITED QUANTITY is unambiguous. "improving",
# "deteriorating", "worsening" are deliberately absent: a deteriorating ratio
# falls and a deteriorating gap rises, so they say nothing about which way the
# figures should move. Bare "up"/"down"/"higher"/"lower" are absent too — "up
# to $80 billion in debt" is not a trend claim, and it is the single most
# common shape in this feed.
_TREND_UP = frozenset({
    "widening", "widened", "rising", "rose", "risen", "growing", "grew",
    "grown", "increasing", "increased", "accelerating", "accelerated",
    "expanding", "expanded", "climbing", "climbed",
})
_TREND_DOWN = frozenset({
    "narrowing", "narrowed", "falling", "fell", "fallen", "declining",
    "declined", "shrinking", "shrank", "shrunk", "contracting", "contracted",
    "dropping", "dropped", "compressing", "compressed",
})

# FY2025 / FY 2025 / H1 2026 / Q3 FY2026 / a bare 2025. The lookahead drops
# the year out of an ISO date (2026-06-30), where the "next number" would be
# the month.
_PERIOD_TOKEN = re.compile(
    r"\b(?:(?:H[12]|Q[1-4])\s+)?(?:FY\s?)?((?:19|20)\d{2})\b(?!\s*[-/]\s*\d)"
)

# Markdown cells and line breaks end a "sentence" as surely as a full stop:
# a table row is not prose, and the words either side of a pipe are not one
# claim. Read without this, one FIG transcript joined a revenue figure from
# a metrics table to a verb from the claim row beneath it.
_SENTENCE_SPLIT = re.compile(r"(?<=[.;!?])\s+|\s*\|\s*|\n+")

# "the FY2026 10-K shows ..." must not offer 10 as FY2026's figure.
_FORM_NAME = re.compile(r"\b(?:10-[KQ]|8-K|S-1|20-F|6-K)\b", re.I)

# A figure ending the text that runs up to a period label belongs to it:
# "5.19x (FY2024)". Anchored at the end and unit-aware, because searching
# for the first number instead found "5.19" inside "5.19x", failed to match
# the text it ended with, and silently fell through to the number on the
# OTHER side of the year.
_TRAILING_VALUE = re.compile(
    r"(?<![\w.%,])-?\d[\d,]*\.?\d*\s*(?:%|x|bp|pp|[KMB]|bn|billion|million|thousand)?\s*$",
    re.I,
)

_SCALE_SUFFIX = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mm": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
    "t": 1e12, "trillion": 1e12,
}
# A magnitude and a percentage are not comparable, and neither is a ratio.
_UNIT_MARKER = re.compile(r"\s*(%|x\b|bp\b|pp\b)", re.I)

# Words that can stand between a trend verb and the figure it is the size OF,
# without another quantity intervening: "declined ~1%", "fell from 5.19x",
# "grew by 47.2%".
_DELTA_FILLER = frozenset({
    "by", "to", "from", "about", "roughly", "approximately", "around",
    "nearly", "another", "over", "just", "some", "a", "an", "of", "at",
})


def _figure_is_the_change_itself(sentence: str, word: str, first_figure: int) -> bool:
    """True when the figures are the SIZE of the move, not levels either side.

    "bookings declined ~1% FY2025 and 2-3% Q3 FY2026" is a deepening decline,
    and reading its figures as levels says the opposite: 1 then 3, rising,
    contradicting "declined". The same shape covers "grew 47.2%", "fell from
    5.19x" and "declined $0.8B FY2024-FY2025". A delta carries its direction
    in the verb, so there is nothing here for this guard to check.
    """
    start = sentence.lower().find(word) + len(word)
    span = sentence[start:first_figure]
    if _PERIOD_TOKEN.search(span):
        return False
    return all(w in _DELTA_FILLER for w in re.findall(r"[a-z]+", span.lower()))


def _value_after(text: str) -> tuple[float, str, str] | None:
    """The first figure in `text`, as (magnitude, unit class, as written).

    The unit class is what stops the comparison reading "41.3%" against
    "$137,791M" as one quantity moving. The magnitude is scale-normalized so
    "$1.35B" and "$1,351M" compare; the third element is the figure as the
    sentence wrote it, because that is what a reader has to find again.
    """
    match = _DEBATE_NUMBER.search(text)
    if match is None:
        return None
    try:
        value = float(match.group(1).replace(",", ""))
    except ValueError:
        return None

    written = match.group(0).strip()
    tail = text[match.end():]
    if match.group(2):                       # the regex's own trailing "%"
        return value, "percent", written

    unit = _UNIT_MARKER.match(tail)
    if unit:
        return value, unit.group(1).lower(), written + unit.group(1)

    scale = re.match(r"\s*([A-Za-z]+)", tail)
    if scale and scale.group(1).lower() in _SCALE_SUFFIX:
        return (
            value * _SCALE_SUFFIX[scale.group(1).lower()],
            "magnitude",
            written + scale.group(1),
        )
    return value, "magnitude", written


def _period_value_pairs(sentence: str) -> list[tuple[int, float, str, str, str]]:
    """(year, magnitude, unit class, period as written, figure as written).

    Both orders occur and both have to work: "FY2025 gap of $832M" puts the
    period first, "fell from 5.19x (FY2024) to 2.63x (FY2025)" puts it after.
    Reading only the first shape mis-paired every figure in the second, which
    is how the first version of this guard flagged a correct AVGO claim.
    """
    pairs: list[tuple[int, float, str, str, str]] = []
    periods = list(_PERIOD_TOKEN.finditer(sentence))
    for index, match in enumerate(periods):
        before = sentence[:match.start()].rstrip(" (")
        trailing = _TRAILING_VALUE.search(before)
        attached = _value_after(trailing.group(0)) if trailing else None
        if attached is None:
            stop = periods[index + 1].start() if index + 1 < len(periods) else len(sentence)
            attached = _value_after(sentence[match.end():stop])
        if attached is None:
            return []
        pairs.append(
            (int(match.group(1)), attached[0], attached[1], match.group(0), attached[2])
        )
    return pairs


def _flag_direction_claims(text: str) -> list[str]:
    """Sentences whose trend word contradicts the figures in the same sentence.

    The gap this closes, found on two NFLX runs a day apart: "a persistent
    OCF/NI gap that is widening — FY2025 gap of $832M versus FY2024's $1,351M
    shortfall". Both figures are correct and both are in the fundamentals
    memo, so `_flag_debate_numbers` cleared them and `check_quotes` had
    nothing to say — and the gap NARROWED. Every other guard here asks where a
    number came from. None reads what the sentence claims the numbers do,
    which is the part a memo's reader acts on.

    Narrow on purpose, and measured rather than assumed: the first version of
    this ran over 36 vault transcripts and returned one true finding and seven
    false ones — a correct AVGO claim whose figures preceded their years, two
    FIG sentences whose trend word governed a different quantity than the one
    the years carried, a markdown table row read as prose, and three risk
    turns where "FY2026 10-K" gave up "10" as a figure. Every condition below
    exists because one of those got through:

      - markdown cells and line breaks END a sentence. A table row is not
        prose and the words either side of a `|` are not one claim.
      - form names (10-K, 10-Q, 8-K, S-1, 20-F) are struck before scanning,
        so "the FY2026 10-K shows" does not offer 10 as FY2026's figure.
      - exactly one direction, from vocabularies that carry one — "improving"
        and "deteriorating" are absent because a deteriorating ratio falls
        while a deteriorating gap rises, and bare "up"/"down" are absent
        because "up to $80 billion" is not a trend claim.
      - the trend word must come BEFORE the figures it is read against. This
        is the one that kills the whole "X fell 8.6% (H1 2026 $141.8M vs H1
        2025 $155.2M) while revenue grew 47.2%" family, where the trailing
        verb belongs to the quantity that has no years attached.
      - exactly two distinct periods, each appearing once, each with a figure,
        the figures sharing a unit class and differing in value.
      - the figures must be LEVELS, not the size of the move. "bookings
        declined ~1% FY2025 and 2-3% Q3 FY2026" is a deepening decline whose
        figures rise; a delta carries its direction in the verb and leaves
        this guard nothing to check. Signed figures are deltas by the same
        argument.

    Everything else is left alone, including trends stated without both
    figures in the same sentence. That silence is the price of a warning a
    reader can trust; the number guard's own history is what a guard costs
    when its flags are usually wrong.
    """
    findings: list[str] = []
    # U+2212 before anything else, exactly as `_flag_debate_numbers` does it:
    # "bookings −1% FY2025" parses as a POSITIVE 1 without this, and the
    # signed-figure rule below never fires on the one shape it exists for.
    scanned = _FORM_NAME.sub("", text.replace("−", "-"))
    for sentence in _SENTENCE_SPLIT.split(scanned):
        words = re.findall(r"[a-z]+", sentence.lower())
        up = [w for w in words if w in _TREND_UP]
        down = [w for w in words if w in _TREND_DOWN]
        if bool(up) == bool(down):          # neither, or both — say nothing
            continue

        pairs = _period_value_pairs(sentence)
        if len(pairs) != 2 or pairs[0][0] == pairs[1][0]:
            continue
        if pairs[0][2] != pairs[1][2] or pairs[0][1] == pairs[1][1]:
            continue

        word = (up or down)[0]
        first_period = _PERIOD_TOKEN.search(sentence)
        if first_period is None or sentence.lower().find(word) > first_period.start():
            continue                        # the verb governs something later
        if min(pairs[0][1], pairs[1][1]) < 0:
            continue                        # a signed figure is a change, not a level
        first_figure = _DEBATE_NUMBER.search(sentence)
        if first_figure and _figure_is_the_change_itself(sentence, word, first_figure.start()):
            continue

        earlier, later = sorted(pairs, key=lambda pair: pair[0])
        actual_up = later[1] > earlier[1]
        if actual_up == bool(up):
            continue

        findings.append(
            f"{word!r} but {later[3]} {later[4]} is "
            f"{'above' if actual_up else 'below'} {earlier[3]} {earlier[4]}"
        )
    return list(dict.fromkeys(findings))


# Formatting, not content: quote characters, whitespace, and the markdown
# markers that carry emphasis rather than meaning ("*", "_", "`").
# Everything with meaning — digits, letters, and the punctuation that changes
# a value (".", ",", "-", ":") — is preserved, so "38.72" still fails to
# match "3.872".
_QUOTE_NOISE = re.compile(r'[\s"\u201c\u201d\u2018\u2019\'*_`]+')


def _norm(text: str) -> str:
    """Project a span onto what a quote actually asserts.

    The technical section of the pack is compact JSON, so the report reads
    `"rsi_14":38.721899422317186` while a debater naturally writes
    `rsi_14: 38.721899422317186` — same field, same value, two characters of
    punctuation apart. Comparing raw, that is a fabricated quote; on the
    first live Haiku turns it flagged 4 claims out of 4, every one of them
    faithful.

    Whitespace goes entirely rather than collapsing to a single space,
    because a quote copied out of a wrapped markdown report carries the wrap.

    Markdown emphasis goes for the same reason, added 2026-08-29 after the
    discrimination probe: the fundamentals report writes "was **not
    effective** as of 2026-06-30" and the debater quoted that sentence with
    the asterisks dropped — a faithful 40-word span reported as a quote not
    in the report, on the one claim the SELL verdict rested on. The markers
    are invisible to a reader and unquotable by construction, so stripping
    them from BOTH sides cannot make a false quote match a real span; it can
    only stop formatting from deciding the answer.
    """
    return _QUOTE_NOISE.sub("", text).lower()


def check_quotes(payload: DebateTurnPayload, texts: dict[str, str]) -> list[str]:
    """claim_ids whose evidence_quote is not actually in the report it names.

    Same class of gap as Phase 4's unverified summary faithfulness, except
    here the fix is whitespace normalization plus `in`. Take the free one.
    """
    return [
        claim.claim_id
        for claim in payload.claims
        if claim.evidence_ref != "none"
        and claim.evidence_quote
        and _norm(claim.evidence_quote) not in _norm(texts.get(claim.evidence_ref, ""))
    ]


def check_concession(
    payload: DebateTurnPayload, turns: list[DebateTurn], side: Side
) -> str | None:
    """Concession must point at a real opposing claim, or it isn't one.
    Returns a dangling `concession_trigger` that was DROPPED, else None.

    Highest-value guardrail in the phase and it costs nothing at runtime: it
    makes "you know, that's a fair point" structurally impossible unless the
    fair point exists in the transcript and belongs to the other side. That
    part is unchanged and is the reason this check must never be relaxed
    into a keyword scan of the argument prose.

    What changed 2026-08-29: `concession_trigger` is no longer gated on
    `stance == 'concede'`. It used to be an error to name a conceded claim on
    any other stance, and the effect of that coupling was that the channel
    NEVER FIRED — across all 42 debate transcripts in the vault, 249 turns
    were `hold` or `sharpen` and not one was `concede`, so every summary the
    pipeline has ever written reported zero concessions.

    The cause was a modeling mismatch, not a weak prompt. `stance` is one
    label per TURN; a concession is about one CLAIM. The shape that actually
    occurs is partial: accept the point you cannot answer, hold the rest.
    MSFT turn 2 (2026-08-29) is the clean instance — "The bear's strongest
    point is the material weakness, and I concede that is a genuine
    overhang. But the authoritative technical relations show..." — and that
    turn is honestly labelled `hold`, because the debater IS holding. The
    old `elif` then made the truthful annotation impossible: naming the
    claim that moved you on a `hold` stance was a hard error that killed the
    turn. The only concession the schema accepted was total capitulation,
    which is not a thing a debater with a case ever does. So the concession
    went into the prose, where the Research Manager read it and cited it
    ("The bull's strongest conceded point is that the material weakness is a
    genuine overhang [turn 2]") while the transcript summary printed
    "Structurally-justified concessions: 0" about the same debate.

    Two severities, matching the two postures already in this module:

      stance='concede'  — RAISES, as before. A full concession changes what
        the transcript says the debate DID, so a trigger naming a claim
        nobody made is corruption, not a typo.
      any other stance  — DROPS AND FLAGS, like `check_rebuts`. A partial
        concession is an annotation on an otherwise sound turn; the stance,
        the argument and the claims are unaffected by a bad pointer, and
        killing a run that has already paid for fundamentals over one is the
        trade e7c82b8 and the `check_rebuts` softening both declined to make.
    """
    prior_ids = {
        claim.claim_id
        for turn in turns
        if turn.side != side
        for claim in turn.payload.claims
    }
    if payload.stance == "concede":
        if payload.concession_trigger not in prior_ids:
            raise ValueError(
                f"concede with concession_trigger="
                f"{payload.concession_trigger!r}, which is not an opposing "
                f"claim_id in this transcript (opposing ids: "
                f"{sorted(prior_ids) or 'none'})"
            )
        return None
    if payload.concession_trigger and payload.concession_trigger not in prior_ids:
        dropped = payload.concession_trigger
        payload.concession_trigger = ""
        return dropped
    return None


def check_rebuts(
    payload: DebateTurnPayload, turns: list[DebateTurn], side: Side
) -> list[str]:
    """Drop rebutted claim_ids that do not belong to a real opposing claim,
    and return the ones dropped so the caller can flag them.

    The completeness gap `check_concession` closed for `stance='concede'`:
    nothing stopped `rebuts` from naming an id that was never made, or one
    belonging to the debater's own side. A turn passing that off would look
    adversarial in the transcript while addressing nothing — the exact
    "theatre" outcome the guardrails in this module exist to rule out.

    Measured 2026-08-23 across all five termination-run transcripts before
    this check existed: 95 of 95 rebutted ids resolved to a claim made in the
    IMMEDIATELY PRECEDING turn — the strongest form of engagement, not just
    "some opposing claim somewhere." This check is deliberately looser than
    that measurement: it accepts any opposing claim so far, matching
    `check_concession`'s scope, because a later round legitimately returns to
    an earlier claim and that should not be an error. What was actually
    observed is stricter than what is enforced; recorded here so the gap
    between them is visible rather than assumed away.
    """
    opposing_ids = {
        claim.claim_id
        for turn in turns
        if turn.side != side
        for claim in turn.payload.claims
    }
    bad = [rid for rid in payload.rebuts if rid not in opposing_ids]
    if bad:
        # DROPPED, not raised. This used to raise ValueError and kill the
        # run. Live cost of that (AVGO, deepseek-v4-flash, 2026-08-29): a
        # bear turn named `technical-contained-uptrend`, an id in no
        # transcript, and took down a run that had already paid for
        # fundamentals, news and technical — $0.1586 for no memo, and no
        # `run_summary` either, since the process died before writing one.
        #
        # A dangling `rebuts` is bad debate hygiene, not a corrupted
        # artifact: the turn's argument and claims are unaffected and remain
        # perfectly usable. Removing the id keeps the transcript honest
        # (nothing downstream can resolve a reference that was never real),
        # and the flag keeps the failure visible. Same posture, and the same
        # reasoning, as e7c82b8's softening of the synthesis fabrication
        # guard: drop the trial, not the run.
        #
        # Why this arose now: the check was added after measuring 95 of 95
        # rebutted ids resolving correctly across five Haiku transcripts.
        # That is a statement about one model. A guard calibrated on one
        # model's failure modes should degrade rather than detonate when a
        # different model deviates.
        payload.rebuts = [rid for rid in payload.rebuts if rid not in bad]
    return bad


def is_productive(payload: DebateTurnPayload, turns: list[DebateTurn]) -> bool:
    """Did this turn introduce a claim_id nobody had used yet?

    OBSERVATIONAL as of 2026-08-24 — see DebateTurn.productive. Still
    computed and still recorded, because it costs nothing and it is still an
    honest reading of a turn; it just no longer feeds the router.
    """
    prior_ids = {
        claim.claim_id for turn in turns for claim in turn.payload.claims
    }
    return bool({claim.claim_id for claim in payload.claims} - prior_ids)


def check_claim_stability(payload: DebateTurnPayload, turns: list[DebateTurn]) -> list[str]:
    """claim_ids in this turn whose text disagrees with their first occurrence.

    Flags, does not raise — a model paraphrasing the same point in different
    words across turns is expected, and rejecting every wording change would
    make claim_id reuse impractical. What this catches is the case that
    matters: two turns using one id for what reads as two different
    assertions, silently, with nothing recording that it happened.

    Compares against the FIRST occurrence specifically (via `canonical_claims`
    on the transcript so far), matching the meaning `canonical_claims` fixes
    for any downstream aggregation — this check and that function agree on
    what a claim_id means, which is the whole point of having both.
    """
    first_by_id = canonical_claims(turns)
    return [
        claim.claim_id
        for claim in payload.claims
        if claim.claim_id in first_by_id
        and first_by_id[claim.claim_id].text != claim.text
    ]


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------

async def _submit(
    client: LLMClient, system_blocks: list[dict], messages: list[dict]
):
    reasoning: dict[str, Any] = {}
    if supports_adaptive_thinking(DEBATE_MODEL):
        reasoning["thinking"] = DEBATE_THINKING
        reasoning["output_config"] = {"effort": DEBATE_EFFORT}

    return await client.messages.create(
        model=DEBATE_MODEL,
        max_tokens=DEBATE_MAX_TOKENS,
        **reasoning,
        system=system_blocks,
        messages=messages,
        tools=[SUBMIT_TOOL],
        # disable_parallel_tool_use, because "call it exactly once" in the
        # prompt is not a constraint. A turn that emitted two submit_argument
        # blocks left the second unanswered on the retry path and 400'd the
        # whole node; forbidding the second call is better than handling it.
        tool_choice={
            "type": "tool",
            "name": "submit_argument",
            "disable_parallel_tool_use": True,
        },
    )



def _assert_within_budget(ticker: str, turns: list[DebateTurn], this_turn: float | None) -> None:
    total = sum(t.estimated_cost_usd or 0.0 for t in turns) + (this_turn or 0.0)
    assert_within_budget(
        total, DEBATE_BUDGET_USD,
        what="debate", context=f" for {ticker}",
        budget=f"per-debate budget after {len(turns) + 1} turn(s)",
        check="DEBATE_MODEL routing and the evidence pack size",
    )


async def run_debate_turn(
    state, side: Side, turn_index: int, client: LLMClient | None = None
) -> DebateTurn:
    """One turn: build the pack, make one forced tool call, run the guards.

    Exactly one retry on a schema violation. Retries inside a node are
    invisible to the checkpointer, so an unbounded retry loop is a runaway
    the round cap CANNOT see — it lives entirely inside one super-step. One
    retry, then raise, then resume from the checkpoint.
    """
    _maybe_crash(turn_index, "before")

    ticker = state["ticker"]
    turns: list[DebateTurn] = list(state.get("debate_turns") or [])
    texts = quotable_texts(state)
    pack = build_evidence_pack(state)
    client = client or get_client(DEBATE_MODEL)

    # Two blocks, stance first. The pack is identical across all six turns,
    # so it caches; the stance prefix differs, so bull and bear keep separate
    # caches — assumed and priced for in the phase-5 estimate.
    system_blocks = [
        {"type": "text", "text": _SYSTEM_BY_SIDE[side]},
        {
            "type": "text",
            "text": pack,
            "cache_control": {"type": "ephemeral"},
        },
    ]
    user_text = (
        f"{render_transcript(turns)}\n\n"
        f"You are the {side.upper()}. This is turn {turn_index} "
        f"(round {(turn_index // 2) + 1}). Submit your argument now."
    )
    messages: list[dict] = [{"role": "user", "content": user_text}]

    usage = UsageSummary()
    payload = await call_with_schema_retry(
        lambda msgs: _submit(client, system_blocks, msgs),
        payload_cls=DebateTurnPayload,
        tool_name="submit_argument",
        messages=messages,
        usage=usage,
        label=f"[debate] {side} turn {turn_index}",
    )

    # Structural guards, and they differ deliberately.
    #
    # `check_concession` still RAISES: a concession is a termination-shaped
    # event, so a concession trigger naming a claim nobody made changes what
    # the transcript says the debate DID, and in a checkpointed graph the
    # last good super-step survives — a loud failure costs a fix-and-resume
    # while silent corruption costs a debate you cannot trust.
    #
    # `check_rebuts` DROPS AND FLAGS: a dangling rebuts id is a bad pointer
    # inside an otherwise sound turn, and killing a run that has already paid
    # for fundamentals over one is a poor trade. See its docstring.
    #
    # `check_concession` does BOTH, split on stance: it raises on a bad
    # trigger under stance='concede' for the reason above, and drops+flags a
    # bad trigger on a partial concession (any other stance), which is the
    # same bad-pointer-in-a-sound-turn shape as `rebuts`.
    dropped_concession = check_concession(payload, turns, side)
    dropped_rebuts = check_rebuts(payload, turns, side)

    node_name = f"{side}_turn"
    event_id = new_event_id(node_name, turn_index=turn_index)
    cost = log_cost(
        ticker,
        f"trading-debate-{side}-r{(turn_index // 2) + 1}",
        usage,
        model=DEBATE_MODEL,
        run_id=state.get("run_id"),
        event_id=event_id,
    )
    _assert_within_budget(ticker, turns, cost)

    turn = DebateTurn(
        turn_index=turn_index,
        round_num=(turn_index // 2) + 1,
        side=side,
        payload=payload,
        productive=is_productive(payload, turns),
        claim_text_drift=check_claim_stability(payload, turns),
        guard_flags=_flag_debate_numbers(
            payload.argument + "\n" + "\n".join(c.text for c in payload.claims),
            pack,
        ),
        unresolved_flags=(
            ([f"unresolved_rebuts: {', '.join(dropped_rebuts)}"] if dropped_rebuts else [])
            + ([f"unresolved_concession: {dropped_concession}"] if dropped_concession else [])
        ),
        direction_flags=_flag_direction_claims(
            payload.argument + "\n" + "\n".join(c.text for c in payload.claims)
        ),
        unquoted_evidence=check_quotes(payload, texts),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        estimated_cost_usd=cost,
        cost_event=record_cost_event(event_id, node_name, usage, DEBATE_MODEL, cost),
    )

    _maybe_crash(turn_index, "after")
    return turn


# ---------------------------------------------------------------------------
# Vault
# ---------------------------------------------------------------------------

def _format_debate_markdown(
    ticker: str, turns: list[DebateTurn], terminated_by: str
) -> str:
    total = sum(t.estimated_cost_usd or 0.0 for t in turns)
    flagged = [f for t in turns for f in t.guard_flags]
    unresolved = [u for t in turns for u in t.unresolved_flags]
    directions = [d for t in turns for d in t.direction_flags]
    unquoted = [c for t in turns for c in t.unquoted_evidence]
    drifted = sorted({cid for t in turns for cid in t.claim_text_drift})
    # Split, because the two mean different things and collapsing them is how
    # this summary came to report "0" about a debate that contained a
    # concession. A `full` concession overturns the debater's position; a
    # `partial` one accepts a specific opposing claim while holding the rest,
    # and is the only shape observed live. Both are structurally justified in
    # exactly the same sense — `check_concession` has verified the named id
    # against the opposing side's claims — so both belong in the count.
    full_concessions = [t for t in turns if t.payload.stance == "concede"]
    partial_concessions = [
        t for t in turns
        if t.payload.stance != "concede" and t.payload.concession_trigger
    ]
    concessions = full_concessions + partial_concessions

    lines = [
        f"# {ticker} — Bull/Bear Debate",
        f"**Turns:** {len(turns)} ({len(turns) // 2} full round(s))",
        f"**Terminated by:** {terminated_by or 'not recorded'}",
        f"**Model:** {DEBATE_MODEL}",
        "",
    ]

    caveats = []
    if not turns:
        caveats.append(
            "**No debate took place.** This ticker's analyst findings carry no "
            "adversarial review, which is not the same as their having survived one."
        )
    if terminated_by == "round_cap":
        caveats.append(
            f"**Truncated.** The debate hit the {len(turns) // 2}-round cap rather "
            f"than resolving — both sides still had new claims when it stopped, so "
            f"this is a truncated argument, not a concluded one."
        )
    if flagged:
        caveats.append(
            f"**{len(flagged)} figure(s) did not appear in any analyst report** and "
            f"may be fabricated: {', '.join(flagged[:10])}. Nothing downstream of "
            f"this debate re-verifies them."
        )
    if unresolved:
        caveats.append(
            f"**{len(unresolved)} turn(s) pointed at something that is not in the "
            f"transcript:** {', '.join(unresolved[:10])}."
        )
    if directions:
        caveats.append(
            f"**{len(directions)} sentence(s) state a direction their own figures "
            f"contradict:** {'; '.join(directions[:5])}. The figures are sourced; "
            f"what is said about them is not."
        )
    if unquoted:
        caveats.append(
            f"**{len(unquoted)} claim(s) cite a report but the quoted span is not "
            f"in it:** {', '.join(unquoted[:10])}."
        )
    if drifted:
        caveats.append(
            f"**{len(drifted)} claim_id(s) were reused with different wording:** "
            f"{', '.join(drifted[:10])}. A claim_id is meant to name one stable "
            f"assertion — read `canonical_claims` (the first occurrence) as the "
            f"authoritative wording, not whichever turn is read last."
        )
    if turns and not concessions:
        caveats.append(
            "**No concession was recorded structurally.** Zero here means no turn "
            "named an opposing `claim_id` in `concession_trigger` — it is NOT "
            "evidence that neither side moved. A debater who concedes a point in "
            "the argument prose without naming its id is not counted, and that has "
            "happened: MSFT 2026-08-29 turn 2 conceded the material weakness in "
            "prose on a `hold` stance, and the Research Manager went on to cite "
            "that concession while this table said zero. Read the arguments before "
            "concluding the debate was unmoved."
        )
    if caveats:
        lines += ["## Caveats", ""] + [f"- {c}" for c in caveats] + [""]

    lines += [
        "## Summary",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Turns | {len(turns)} |",
        f"| Structurally-justified concessions | {len(concessions)} "
        f"({len(full_concessions)} full, {len(partial_concessions)} partial) |",
        f"| Unproductive turns (no new claim, observational only) | "
        f"{sum(1 for t in turns if not t.productive)} |",
        f"| Flagged figures | {len(flagged)} |",
        f"| Unresolved references | {len(unresolved)} |",
        f"| Contradicted directions | {len(directions)} |",
        f"| Unverified quotes | {len(unquoted)} |",
        f"| Reused claim_ids with drifted text | {len(drifted)} |",
        f"| Estimated cost | ${total:.4f} |",
        "",
        "## Transcript",
        "",
    ]

    for turn in turns:
        lines += [
            f"### Turn {turn.turn_index} — {turn.side.upper()} (round {turn.round_num})",
            f"*stance:* `{turn.payload.stance}`"
            + (
                f" · *concedes to:* `{turn.payload.concession_trigger}`"
                + ("" if turn.payload.stance == "concede" else " (partial)")
                if turn.payload.concession_trigger
                else ""
            )
            + (f" · *rebuts:* {', '.join(turn.payload.rebuts)}" if turn.payload.rebuts else "")
            + (" · **unproductive**" if not turn.productive else ""),
            "",
            turn.payload.argument,
            "",
            "| Claim | Source | Assertion | Quote |",
            "|---|---|---|---|",
        ]
        for claim in turn.payload.claims:
            text = claim.text.replace("|", "\\|")
            quote = claim.evidence_quote.replace("|", "\\|")
            marker = " ⚠︎" if claim.claim_id in turn.claim_text_drift else ""
            lines.append(
                f"| `{claim.claim_id}`{marker} | {claim.evidence_ref} | {text} | {quote} |"
            )
        if turn.guard_flags:
            lines.append("")
            lines.append(f"*Flagged figures:* {', '.join(turn.guard_flags)}")
        if turn.unresolved_flags:
            lines.append(f"*Unresolved references:* {', '.join(turn.unresolved_flags)}")
        if turn.direction_flags:
            lines.append(f"*Contradicted direction:* {'; '.join(turn.direction_flags)}")
        if turn.unquoted_evidence:
            lines.append(f"*Unverified quotes:* {', '.join(turn.unquoted_evidence)}")
        if turn.claim_text_drift:
            lines.append(
                f"*⚠︎ Reused with different wording than the first occurrence:* "
                f"{', '.join(turn.claim_text_drift)}"
            )
        lines.append("")

    return "\n".join(lines)


def save_debate_transcript(
    ticker: str,
    turns: list[DebateTurn],
    terminated_by: str,
    provenance: str | None = None,
) -> Path:
    content = _format_debate_markdown(ticker.upper(), turns, terminated_by)
    total = sum(t.estimated_cost_usd or 0.0 for t in turns)
    return _save_output(
        content,
        ticker.upper(),
        "debate",
        cost_usd=total if turns else None,
        provenance=provenance,
        model=DEBATE_MODEL,
    )
