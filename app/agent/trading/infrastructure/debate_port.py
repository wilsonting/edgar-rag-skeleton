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
"""

from __future__ import annotations

import copy
import os
import re
import sys
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from pydantic import ValidationError

from app.agent.researcher import (
    AGENT_MODEL,
    _MODEL_PRICING,
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
DEBATE_MODEL = os.getenv("TRADING_DEBATE_MODEL") or AGENT_MODEL

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


if DEBATE_MODEL not in _MODEL_PRICING:
    # Not fatal, but the budget assertion is the only thing standing between a
    # prompt-bloat regression and an unbounded bill, and an unpriced model
    # makes every turn cost None — which sums to 0.00 and can never trip it.
    # Say so once at import rather than letting the ceiling be silently
    # absent for a whole run.
    print(
        f"[debate] WARNING: no pricing configured for {DEBATE_MODEL} — per-turn "
        f"costs will log as null and the ${DEBATE_BUDGET_USD:.2f} budget "
        f"assertion cannot fire. Add it to _MODEL_PRICING in researcher.py."
    )

# Forced-failure hooks for the resume tests. Deliberately in the port rather
# than the node: variant B has to die AFTER the API call and before the node
# returns, which is the window a `kill -9` would land in and the one the
# add-reducer's re-execution behaviour is actually tested by.
_CRASH_AT = os.getenv("DEBATE_CRASH_AT_TURN")
_CRASH_WHEN = os.getenv("DEBATE_CRASH_WHEN", "before")   # "before" | "after"


def _maybe_crash(turn_index: int, when: str) -> None:
    """os._exit, not sys.exit or raise. Both of those unwind cleanly and let
    the framework write a shutdown checkpoint, which is a strictly easier
    scenario than the process kill the exit criterion describes."""
    if _CRASH_AT is None or int(_CRASH_AT) != turn_index or _CRASH_WHEN != when:
        return
    print(f"[debate] FORCED CRASH {when} turn {turn_index} (DEBATE_CRASH_AT_TURN)")
    # os._exit skips every atexit hook and every buffered stream, so without
    # this the crash message and the whole run's node progress are lost —
    # which is a faithful simulation of kill -9 and a useless test log.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


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
5. Stance:
   - 'hold'    — you are maintaining your position against the rebuttal.
   - 'sharpen' — you are narrowing or qualifying your own earlier claim.
   - 'concede' — you accept a specific opposing claim. Then, and only then,
     `concession_trigger` must name the opponent claim_id that moved you; on
     every other stance it is the literal string 'none'. Naming a claim on a
     non-concede stance is an error, and so is conceding to a claim_id that
     is not in the transcript.

A report marked "NOT RUN" is missing evidence, not neutral evidence. Do not
infer anything from its absence, and do not argue from it in either direction.

Do not concede to be agreeable, and do not manufacture disagreement. If the
evidence genuinely does not support your side on a point, 'concede' it and
argue the points where it does.

Call `submit_argument` exactly once. Say nothing else."""

BULL_SYSTEM = _SYSTEM_TEMPLATE.replace("{STANCE}", BULL_STANCE)
BEAR_SYSTEM = _SYSTEM_TEMPLATE.replace("{STANCE}", BEAR_STANCE)

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

    Flags, never blocks. This guard is new enough to have unknown
    false-positive classes, and debate prose is looser than the templated
    technical interpretation, so expect more of them here than in Phase 3.
    """
    text = text.replace("−", "-")
    pack = evidence_pack.replace("−", "-").replace(",", "")

    pack_tokens = {m.group(1) for m in _DEBATE_NUMBER.finditer(pack)}
    known: list[float] = []
    for token in pack_tokens:
        try:
            known.append(float(token))
        except ValueError:
            continue

    # Second opinion, consulted ONLY for percent forms (see docstring).
    tolerance_flags = set(_flag_unmatched_numbers_against(text, known))

    scanned = _PERIOD_LABEL.sub("", text)
    flagged: list[str] = []
    for match in _DEBATE_NUMBER.finditer(scanned):
        raw, percent = match.group(1), match.group(2)
        if raw.replace(",", "") in pack_tokens:
            continue
        if _is_rounding_of(raw, known):
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


def _is_rounding_of(raw: str, known: list[float]) -> bool:
    """True when some pack value rounds to `raw` at `raw`'s own precision.

    Precision-scoped on purpose. A fixed tolerance widens the guard's blind
    spot as the pack grows; this one does not — "41.2" only ever clears
    against a value in [41.15, 41.25), whatever else is in the pack.
    """
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return False
    fraction = raw.split(".")
    places = len(fraction[1]) if len(fraction) == 2 else 0
    return any(round(kv, places) == value for kv in known)


# Formatting, not content: quote characters and whitespace. Everything with
# meaning — digits, letters, and the punctuation that changes a value
# (".", ",", "-", ":") — is preserved, so "38.72" still fails to match
# "3.872".
_QUOTE_NOISE = re.compile(r'[\s"\u201c\u201d\u2018\u2019\']+')


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
) -> None:
    """Concession must point at a real opposing claim, or it isn't one.

    Highest-value guardrail in the phase and it costs nothing at runtime: it
    makes "you know, that's a fair point" structurally impossible unless the
    fair point exists in the transcript and belongs to the other side.
    """
    if payload.stance == "concede":
        prior_ids = {
            claim.claim_id
            for turn in turns
            if turn.side != side
            for claim in turn.payload.claims
        }
        if payload.concession_trigger not in prior_ids:
            raise ValueError(
                f"concede with concession_trigger="
                f"{payload.concession_trigger!r}, which is not an opposing "
                f"claim_id in this transcript (opposing ids: "
                f"{sorted(prior_ids) or 'none'})"
            )
    elif payload.concession_trigger:
        raise ValueError(
            f"concession_trigger={payload.concession_trigger!r} set on a "
            f"non-concede stance ({payload.stance})"
        )


def check_rebuts(payload: DebateTurnPayload, turns: list[DebateTurn], side: Side) -> None:
    """Every rebutted claim_id must belong to a real opposing claim.

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
        raise ValueError(
            f"rebuts names {bad!r}, which {'is' if len(bad) == 1 else 'are'} "
            f"not opposing claim_id(s) in this transcript (opposing ids: "
            f"{sorted(opposing_ids) or 'none'}) — either a hallucinated id or "
            f"the debater's own side"
        )


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
    client: AsyncAnthropic, system_blocks: list[dict], messages: list[dict]
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


def _accumulate(usage: UsageSummary, raw) -> None:
    usage.input_tokens += raw.input_tokens
    usage.cache_write_tokens += raw.cache_creation_input_tokens or 0
    usage.cache_read_tokens += raw.cache_read_input_tokens or 0
    usage.output_tokens += raw.output_tokens


def _tool_block(response):
    return next((b for b in response.content if b.type == "tool_use"), None)


def _extract(response) -> DebateTurnPayload:
    """Raises ValidationError on anything the retry can correct — including a
    response with no tool call at all, which is the same class of failure as
    a malformed one and gets the same single retry."""
    block = _tool_block(response)
    if block is None:
        raise ValidationError.from_exception_data(
            "DebateTurnPayload",
            [{"type": "missing", "loc": ("submit_argument",), "input": None}],
        )
    return DebateTurnPayload.model_validate(block.input)


_CORRECTION = (
    "That submission did not validate:\n{error}\n\n"
    "Call submit_argument once more, correcting exactly those fields. "
    "Change nothing else."
)


def _retry_messages(messages: list[dict], response, error: Exception) -> list[dict]:
    """Feed the validation error back in a shape the API accepts.

    A `tool_use` block MUST be answered by a `tool_result` in the next
    message — appending a plain user turn after one is a 400, which is how
    this was found. When the model returned no tool call there is nothing to
    answer, so the correction goes back as an ordinary user turn instead.
    """
    turns = list(messages)
    if response.content:
        turns.append({"role": "assistant", "content": response.content})

    correction = _CORRECTION.format(error=error)
    # EVERY tool_use block, not just the one that was validated. The API
    # requires a tool_result per tool_use; answering only the first is the
    # same 400 in a different disguise.
    results = [
        {
            "type": "tool_result",
            "tool_use_id": block.id,
            "is_error": True,
            "content": correction,
        }
        for block in response.content
        if block.type == "tool_use"
    ]
    turns.append({"role": "user", "content": results or correction})
    return turns


def _assert_within_budget(ticker: str, turns: list[DebateTurn], this_turn: float | None) -> None:
    """Fires as soon as the running total crosses the ceiling, not at the end.

    An assertion cannot refund a turn already paid for, so the earliest
    possible turn is the only useful place for it. What it actually catches
    is prompt bloat or a runaway — the round cap bounds normal spend.
    """
    total = sum(t.estimated_cost_usd or 0.0 for t in turns) + (this_turn or 0.0)
    if total > DEBATE_BUDGET_USD:
        raise AssertionError(
            f"debate cost ${total:.4f} for {ticker} exceeds the "
            f"${DEBATE_BUDGET_USD:.2f} per-debate budget after "
            f"{len(turns) + 1} turn(s) — check DEBATE_MODEL routing and the "
            f"evidence pack size before rerunning"
        )


async def run_debate_turn(
    state, side: Side, turn_index: int, client: AsyncAnthropic | None = None
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
    client = client or AsyncAnthropic()

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
    response = await _submit(client, system_blocks, messages)
    _accumulate(usage, response.usage)
    try:
        payload = _extract(response)
    except ValidationError as first:
        block = _tool_block(response)
        print(
            f"[debate] {side} turn {turn_index}: schema violation, one retry "
            f"— stop_reason={response.stop_reason} "
            f"keys={sorted(block.input) if block else None} "
            f"— {'; '.join(str(first).splitlines()[1:5])}"
        )
        messages = _retry_messages(messages, response, first)
        retry = await _submit(client, system_blocks, messages)
        _accumulate(usage, retry.usage)
        payload = _extract(retry)   # a second failure raises out of the node

    # Structural guards. These raise: in a checkpointed graph the last good
    # super-step survives, so a loud failure costs a fix-and-resume while
    # silent corruption costs a debate you cannot trust.
    check_concession(payload, turns, side)
    check_rebuts(payload, turns, side)

    cost = log_cost(
        ticker,
        f"trading-debate-{side}-r{(turn_index // 2) + 1}",
        usage,
        model=DEBATE_MODEL,
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
        unquoted_evidence=check_quotes(payload, texts),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        estimated_cost_usd=cost,
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
    unquoted = [c for t in turns for c in t.unquoted_evidence]
    drifted = sorted({cid for t in turns for cid in t.claim_text_drift})
    concessions = [t for t in turns if t.payload.stance == "concede"]

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
    if caveats:
        lines += ["## Caveats", ""] + [f"- {c}" for c in caveats] + [""]

    lines += [
        "## Summary",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Turns | {len(turns)} |",
        f"| Structurally-justified concessions | {len(concessions)} |",
        f"| Unproductive turns (no new claim, observational only) | "
        f"{sum(1 for t in turns if not t.productive)} |",
        f"| Flagged figures | {len(flagged)} |",
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
