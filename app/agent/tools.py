"""
Tool schemas and dispatch for the research agent.
"""

import ast
import logging
import operator
import os
import re
import sys

import httpx
from pydantic import ValidationError

from app.domain.token_usage import USAGE_HEADER, TokenUsage


# Base URL of your running FastAPI server. Override when you wire step 2.
logger = logging.getLogger(__name__)

API_BASE = "http://localhost:8000"
HTTP_TIMEOUT = 300.0  # ingestion can be slow; give it room

# `ask_edgar` is the most expensive thing the agent can do and nothing
# bounded it. Measured on the Phase 9 battery: NFLX 22 calls, AVGO 40, ACN
# 40, every question distinct (checked -- there is no duplicate work to
# dedupe away). At ~$0.008 of server-side spend per call plus a full context
# round-trip, the call count is the single biggest cost lever in a
# fundamentals run.
#
# The budget is announced to the agent rather than sprung on it. A blind cap
# truncates wherever the agent happens to be when it trips, which on a
# 12-item checklist means the last items silently get nothing; an announced
# one lets it allocate. That is the same reasoning behind telling a person a
# deadline at the start rather than at the end.
#
# 30 is a judgement, not a measurement: above NFLX's 22, below the 40 that
# AVGO and ACN each used. ACN completed its checklist in 40, so this WILL
# bind on dense tickers -- deliberately, since the alternative is an
# unbounded cost per run. Raise it with ASK_EDGAR_MAX_CALLS if a ticker's
# analysis is being cut short in a way that matters.
ASK_EDGAR_MAX_CALLS = int(os.getenv("ASK_EDGAR_MAX_CALLS", "30"))
# Chunks retrieved per ask_edgar call, and ~77% of a call's input cost:
# each chunk averages ~610 tokens, so 8 -> 5 would save ~1,800 tokens/call,
# ~$0.05 per fundamentals run at 30 calls.
#
# 5 WAS TRIED AND REVERTED (2026-08-27). Run
# `scripts/probe_retrieval_rank_decay.py` before touching this -- it replays
# the real questions from a battery. Over 102 of them:
#
#     rank 1 mean similarity 0.6799 ... rank 8 mean similarity 0.6451
#     rank 1 -> rank 8 decay 5.1%       rank 5 -> rank 6 drop 0.5%
#     ranks 6-8 supply a section_path absent from ranks 1-5: 59% of questions
#
# There is no cliff at 5. The tail is nearly as relevant as the head, and on
# a majority of questions it carries filing sections nothing else retrieved
# -- which is what a cross-section forensic checklist exists to read. The
# five cents is real; so is the coverage, and the coverage is worth more.
#
# Sent explicitly rather than leaning on the /ask endpoint's own default, so
# the agent's retrieval width cannot silently track an unrelated API default.
ASK_EDGAR_K = int(os.getenv("ASK_EDGAR_K", "8"))
# How many calls out from the cap the agent starts being told to wrap up.
_ASK_EDGAR_WARN_AT = 5
_ASK_EDGAR_CALLS = 0

# Results by normalized expression, for the run. Whitespace only: two
# expressions that differ in spacing are the same computation, while two
# that differ in SCALE ("45183.036 - 39001.0" vs "45183036 - 39001000") are
# deliberately kept apart even though they reduce to the same ratio — a
# cache is not the place to assert that two differently-written derivations
# are equivalent.
_CALC_CACHE: dict[str, str] = {}
_CALC_WHITESPACE = re.compile(r"\s+")


def _normalize_expression(expression: str) -> str:
    return _CALC_WHITESPACE.sub("", expression)


# ---------------------------------------------------------------------------
# Tool schemas — sent to the model so it knows what it can call.
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "check_corpus",
        "description": (
            "Check what filings are available for a ticker before asking "
            "questions. Always call this first for any ticker. Returns filing "
            "count, date range, and chunk counts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "ingest_ticker",
        "description": (
            "Ingest SEC filings for a ticker not in the corpus or lacking "
            "history. Slow: 30-60s per filing. Call once with limit=3. "
            "Omit form_type to auto-detect the filer's form family: 10-K "
            "for a domestic filer, or 20-F for a foreign private issuer "
            "(e.g. ASML) — those never file a 10-K. Pass form_type "
            "explicitly for a specific type: '10-Q'/'8-K' (domestic) or "
            "'6-K' (foreign private issuer's interim/current report)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "limit": {"type": "integer"},
                "form_type": {
                    "type": "string",
                    "description": (
                        "Filing type to ingest: '10-K', '10-Q', '8-K', "
                        "'20-F', or '6-K'. Omit to auto-detect."
                    ),
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "ask_edgar",
        "description": (
            "Ask one specific question about SEC filings. Returns an answer "
            "with citations and source excerpts. Best for cross-section "
            "analysis, year-over-year comparisons, risk factors, MD&A "
            "commentary, and segment breakdowns.\n\n"
            f"BUDGETED: you may call this at most {ASK_EDGAR_MAX_CALLS} times "
            "in one analysis, and it is the most expensive tool available to "
            "you. Plan the whole checklist against that number before "
            "spending the first call — ask one broad question that covers "
            "several checklist items rather than one narrow question per "
            "item, and do not re-ask something an earlier answer already "
            "told you. When the budget runs out you will be told to write "
            "the memo from what you have."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "tickers": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question"],
        },
    },
    {
        "name": "extract_metrics",
        "description": (
            "Extract structured financial metrics (revenue, gross margin, "
            "FCF, SBC%, net dollar retention) for a ticker and filing period."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "fiscal_period": {"type": "string", "description": "e.g. 'FY2025' or 'Q1 2026'"},
                "filing_type": {"type": "string", "description": "e.g. '10-K' or '10-Q'"},
                "filed_date": {"type": "string", "format": "date"},
                "filed_after": {"type": "string", "format": "date"},
                "filed_before": {"type": "string", "format": "date"},
            },
            "required": ["ticker", "fiscal_period", "filing_type", "filed_date"],
        },
    },
    {
        "name": "check_latest_filings",
        "description": (
            "Check SEC EDGAR for the latest filings for a ticker and "
            "compare with what is already in the corpus. Auto-detects "
            "whether the filer uses domestic forms (10-K/10-Q/8-K) or is a "
            "foreign private issuer using 20-F/6-K instead — the response's "
            "form_types_searched field states which family was checked, so "
            "a zero total_on_sec with domestic form types searched does NOT "
            "mean the company has no SEC filings at all. Returns which "
            "filings are new and not yet ingested. Use this to ensure the "
            "corpus has the most recent reports before running analysis.\n\n"
            "By default this returns PERIODIC reports only (10-K/10-Q, or "
            "20-F for foreign private issuers) — the financial statements a "
            "checklist is built from. Event filings (8-K/6-K) are excluded "
            "because they outnumber the periodic ones several times over and "
            "would fill your context without informing the analysis. Pass "
            "form_types explicitly, e.g. [\"8-K\"], on the rare occasion you "
            "need a specific event filing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "form_types": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate a mathematical expression. Use for EVERY ratio, growth "
            "rate, margin and percentage — never compute one yourself. Each "
            "numeric literal in the expression must be a figure you retrieved "
            "verbatim from a filing in this session, and must be declared in "
            "`inputs` with its fiscal period and source citation. Do not "
            "reconstruct figures with arithmetic (no 27.6*1000): retrieve the "
            "exact value instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "e.g. '(32667.3 - 28262.9) / 28262.9 * 100'. Every "
                        "figure must appear exactly as the filing states it."
                    ),
                },
                "inputs": {
                    "type": "array",
                    "description": (
                        "One entry per retrieved figure used in the expression. "
                        "Scalars that are part of the operation itself (100 for "
                        "percent, 2 for a square root) do not need declaring."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "number"},
                            "label": {"type": "string", "description": "e.g. 'total net sales'"},
                            "fiscal_period": {"type": "string", "description": "e.g. 'FY2024'"},
                            "source": {"type": "string", "description": "e.g. 'ASML 20-F 2026 §Item 6'"},
                            "unit": {
                                "type": "string",
                                "enum": ["ones", "thousands", "millions", "billions", "percent", "ratio"],
                                "description": (
                                    "The scale this figure is actually reported at in the "
                                    "filing — e.g. a balance-sheet line in thousands and a "
                                    "segment-table line in millions both appear, unconverted, "
                                    "in the same 10-K. Declare the real scale here instead of "
                                    "converting by hand in the expression; the tool normalizes "
                                    "before combining values reported at different scales."
                                ),
                            },
                        },
                        "required": ["value", "label", "fiscal_period", "source", "unit"],
                    },
                },
            },
            "required": ["expression", "inputs"],
        },
    }
]


# ---------------------------------------------------------------------------
# Safe calculator — AST-walking evaluator, not eval(). Permits only numeric
# literals and + - * / ** and unary minus. Anything else (calls, names,
# attribute access) raises.
# ---------------------------------------------------------------------------

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


# Scale factors for each unit a `calculate` input can declare. A figure's
# literal in the expression is normalized by its declared unit before the
# arithmetic runs, so a thousands-scale figure and a millions-scale figure
# combine correctly even though the model never wrote a conversion factor.
_UNIT_SCALES = {
    "ones": 1.0,
    "thousands": 1_000.0,
    "millions": 1_000_000.0,
    "billions": 1_000_000_000.0,
    "percent": 1.0,
    "ratio": 1.0,
}


def safe_calculate(expression: str, inputs: list[dict] | None = None) -> str:
    try:
        node = ast.parse(expression, mode="eval").body
        scale_by_value = _scale_by_value(inputs or [])
        return str(_eval_node(node, scale_by_value))
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


def _scale_by_value(inputs: list[dict]) -> dict[float, float]:
    out: dict[float, float] = {}
    for i in inputs:
        if i.get("value") is None:
            continue
        scale = _UNIT_SCALES.get(i.get("unit"))
        if scale is not None:
            out[float(i["value"])] = scale
    return out


def _eval_node(node, scale_by_value: dict[float, float] | None = None):
    scale_by_value = scale_by_value or {}
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            value = float(node.value)
            return value * scale_by_value.get(value, 1.0)
        raise ValueError("only numeric constants allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](
            _eval_node(node.left, scale_by_value), _eval_node(node.right, scale_by_value)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand, scale_by_value))
    raise ValueError(f"disallowed expression element: {type(node).__name__}")


def _strictify(schema: dict) -> dict:
    """Rewrite a schema into the form strict tool-calling requires.

    Both dialects want the same three things, recursively: every object
    closed with `additionalProperties: false`, every declared property
    listed in `required`, and anything genuinely optional expressed as
    nullable rather than absent.

    Done as a transformation rather than by hand-editing the schemas above
    so the readable version stays the source of truth — the `required` list
    there still says which arguments actually matter, and this only
    restates it in the shape the API enforces.

    Why bother when `_validate_tool_inputs` already catches a bad call: the
    validator recovers from the mistake, strict prevents it. The run that
    exposed all of this lost 376 seconds to one mis-named argument, and a
    recovered tool call still costs a turn against LOOP_MAX_TURNS — which
    the priciest runs already exhaust.

    An optional property becomes `["<type>", "null"]`, and the dispatch
    code reads those through `.get()`, so an explicit null behaves exactly
    as the previously-absent key did.
    """
    if schema.get("type") != "object":
        return schema

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    out_properties = {}

    for name, prop in properties.items():
        prop = dict(prop)
        if prop.get("type") == "object":
            prop = _strictify(prop)
        elif prop.get("type") == "array" and isinstance(prop.get("items"), dict):
            prop["items"] = _strictify(prop["items"])
        if name not in required and "type" in prop and not isinstance(prop["type"], list):
            prop["type"] = [prop["type"], "null"]
        out_properties[name] = prop

    return {
        **schema,
        "properties": out_properties,
        "required": list(properties),
        "additionalProperties": False,
    }


# Applied after the schemas above are declared, and `strict` set alongside.
# Kept together so a tool added later cannot pick up one without the other:
# `strict: true` on a schema that is not strict-compatible is a 400 on the
# first call, not a quiet degradation.
# Captured BEFORE the rewrite. `_strictify` lists every property in
# `required` because that is what strict mode demands, but that is a wire
# format, not the tool's real contract: `form_types` is still optional, and
# a call that simply omits it must not be rejected as incomplete by our own
# validator. So validation keeps asking the original question.
_REQUIRED_BY_NAME = {
    tool["name"]: list(tool["input_schema"].get("required", [])) for tool in TOOLS
}

for _tool in TOOLS:
    _tool["input_schema"] = _strictify(_tool["input_schema"])
    _tool["strict"] = True

# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

_SCHEMA_BY_NAME = {tool["name"]: tool["input_schema"] for tool in TOOLS}


def _validate_tool_inputs(name: str, inputs: dict) -> str | None:
    """Check a tool call's arguments against its own schema before dispatch.

    Returns an error message for the model, or None when the call is fine.

    Exists because a mis-named argument used to kill the whole run. Every
    other failure inside `_dispatch` comes back as a STRING the model can
    read and react to — a non-200 from the API, a rejected expression — but
    the argument reads were bare subscripts, so `check_latest_filings`
    called with `tickers=[...]` instead of `ticker="..."` raised a KeyError
    that propagated out of `execute_tool` and ended a 376-second run at the
    tool call. Nothing about that is provider-specific; it had simply never
    been the failing model's mistake before.

    The plural/singular split is the trap that sprang it: `ask_edgar` takes
    `tickers` (a list) and every other tool takes `ticker` (a string), so a
    model generalising from one call to the next gets it wrong in a way no
    amount of prompt wording reliably prevents. Naming the accepted
    arguments back to the model is what makes the next attempt succeed.

    Unknown arguments are reported, not ignored: a call carrying `tickers`
    would otherwise fail the `ticker` check with no hint about the list it
    did send, and the model would have to guess what it got wrong.
    """
    schema = _SCHEMA_BY_NAME.get(name)
    if schema is None:
        return None

    properties = schema.get("properties", {})
    required = _REQUIRED_BY_NAME.get(name, [])
    missing = [k for k in required if k not in inputs]
    unknown = [k for k in inputs if k not in properties]
    if not missing and not unknown:
        return None

    problems = []
    if missing:
        problems.append(f"missing required argument(s): {', '.join(sorted(missing))}")
    if unknown:
        problems.append(f"unexpected argument(s): {', '.join(sorted(unknown))}")

    accepted = ", ".join(
        f"{k} ({v.get('type', 'any')})" + ("" if k in required else ", optional")
        for k, v in properties.items()
    )
    return (
        f"Error: {name} was called with " + "; ".join(problems) + ". "
        f"This tool accepts exactly: {accepted}. "
        f"Re-issue the call with the correct argument names."
    )

# ---------------------------------------------------------------------------
# Dispatch. Each branch prints its call/result so you can watch the agent
# reason. Replace the stub branches in step 2.
# ---------------------------------------------------------------------------

# Toggle to False in step 2 once the HTTP branches are wired and tested.
USE_STUBS = False


async def execute_tool(name: str, inputs: dict) -> str:
    print(f"  [tool call] {name}({inputs})")
    record_log_line(f"  [tool call] {name}({inputs})")

    # Before dispatch, and returned WITHOUT recording into the provenance
    # corpus: a rejected call retrieved nothing, and the corpus is what
    # every numeric guard checks memo figures against.
    problem = _validate_tool_inputs(name, inputs)
    if problem:
        print(f"  [tool result] {problem}", file=sys.stderr)
        record_log_line(f"  [tool result] {problem}")
        return problem

    result = await _dispatch(name, inputs)

    # Feed tool output into the provenance corpus so calculate() can verify
    # its inputs were actually retrieved. Exclusions:
    #   - calculate itself: computed results must not count as retrieved,
    #     or derived figures launder themselves one call at a time
    #   - the verifier's warning block: a flagged figure appears inside the
    #     answer text, and recording it would make the fabrication count
    #     as "returned by a tool" for later calculate calls
    #   - similarity scores in ask_edgar citation lines: retrieval
    #     diagnostics, not filing figures. Left in, every sim=0.XXX becomes
    #     a corpus number the verifier's scale-tolerant fallback can match
    #     a fabricated memo figure against (516.5/1000 ≈ sim=0.516)
    if name != "calculate":
        clean = result.split("WARNING — these figures")[0]
        clean = re.sub(r"sim=\d\.\d+", "sim=", clean)
        record_tool_output(clean)

    # Truncate noisy results in the console; the model still gets the full
    # text, and so does the session log — a 300-char preview would gut the
    # saved trace's audit value (tracing a memo figure back to the exact
    # tool output that returned it is the whole point of the file)
    preview = result if len(result) < 300 else result[:300] + " …"
    print(f"  [tool result] {preview}", file=sys.stderr)
    record_log_line(f"  [tool result] {result}")
    return result


async def _dispatch(name: str, inputs: dict) -> str:
    if name == "calculate":
        expression = inputs["expression"]
        # Validation runs on EVERY call, cache hit or not. It checks the
        # supplied `inputs` have provenance, and a repeat can arrive with
        # different — possibly worse — inputs than the call that populated
        # the cache. Short-circuiting before this would let the second call
        # launder the first one's provenance.
        err = validate_calculate_inputs(expression, inputs.get("inputs", []))
        if err:
            record_rejected_calc(expression, inputs.get("inputs", []), err)
            return err

        key = _normalize_expression(expression)
        if key in _CALC_CACHE:
            # Note honestly what this does and does not save. `calculate` is
            # pure Python with no API call, so the direct cost of a repeat is
            # ~zero and was already paid before this function ran: the
            # expensive part is the agent TURN, a full context round-trip
            # (~22k cache-read tokens, ~$0.0022) spent to reach this line.
            # Memoisation cannot refund that. What it can do is tell the
            # agent it is repeating itself, which is a behavioural nudge
            # against the NEXT duplicate. Measured cause: NFLX ran
            # "10149273 - 688220" three times and one growth-rate expression
            # twice in a single run (Phase 9 cost audit).
            #
            # The note carries no digits of its own, so it adds nothing to
            # the provenance corpus the containment guards scan.
            return (
                f"{_CALC_CACHE[key]}  [already computed earlier this run — "
                f"identical expression, identical result. Check your earlier "
                f"working before re-deriving a figure.]"
            )

        result = safe_calculate(expression, inputs.get("inputs", []))
        record_calc_result(result)
        _CALC_CACHE[key] = result
        return result

    # Budget check BEFORE any transport is set up. A refused call has to
    # cost nothing at all -- that is the whole point of it -- and putting
    # the check inside the HTTP context meant a refusal still built a
    # client. Caught by test_the_refusal_makes_no_http_call.
    if name == "ask_edgar":
        global _ASK_EDGAR_CALLS
        if _ASK_EDGAR_CALLS >= ASK_EDGAR_MAX_CALLS:
            # Refuse rather than raise: the agent's correct response is to
            # write the memo from what it has, exactly as it does at
            # MAX_TURNS. An exception would lose the whole run's work over a
            # budget that is a preference, not a failure.
            return (
                f"BUDGET EXHAUSTED: you have used all "
                f"{ASK_EDGAR_MAX_CALLS} of your ask_edgar calls for this "
                f"analysis. No further filing queries are available. Write "
                f"the memo now from what you have already gathered, and "
                f"record any checklist item you could not complete as an "
                f"explicit data gap rather than leaving it unmentioned."
            )
        _ASK_EDGAR_CALLS += 1

    if USE_STUBS:
        return _stub(name, inputs)

    # STEP 2: real HTTP calls. Un-stub by setting USE_STUBS = False and
    # confirming each endpoint below matches your FastAPI routes.
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as http:
        if name == "check_corpus":
            # STEP 2: confirm this route/param exists, or add it to main.py
            resp = await http.get(
                f"{API_BASE}/corpus-status", params={"ticker": inputs["ticker"]}
            )
            if resp.status_code != 200:
                return f"Error from /check_corpus: {resp.status_code} — {resp.text[:500]}"

            return resp.text

        if name == "ingest_ticker":
            payload = {"ticker": inputs["ticker"], "limit": inputs.get("limit", 3)}
            if "form_type" in inputs:
                payload["form_type"] = inputs["form_type"]
            resp = await http.post(f"{API_BASE}/ingest", json=payload)
            if resp.status_code != 200:
                return f"Error from /ingest_ticker: {resp.status_code} — {resp.text[:500]}"

            return resp.text

        if name == "ask_edgar":
            resp = await http.post(
                f"{API_BASE}/ask",
                json={
                    "question": inputs["question"],
                    "tickers": inputs.get("tickers"),
                    "k": ASK_EDGAR_K,
                },
            )
            if resp.status_code != 200:
                return f"Error from /ask: {resp.status_code} — {resp.text[:500]}"
            _record_delegated_usage(resp)

            data = resp.json()
            citations = "\n".join(
                f"  [{c['citation']}] sim={c['similarity']:.3f}"
                for c in data.get("chunks", [])
            )
            out = f"{data['answer']}\n\nSources:\n{citations}"
            remaining = ASK_EDGAR_MAX_CALLS - _ASK_EDGAR_CALLS
            if remaining <= _ASK_EDGAR_WARN_AT:
                # Only inside the warn band. Appending a counter to all 30
                # answers would put a changing number into every tool result,
                # and tool results are the agent's provenance corpus -- the
                # containment guards scan it for figures that "back" memo
                # claims. Five lines of budget text is a cost worth paying;
                # thirty is not.
                out += (
                    f"\n\n[BUDGET] {remaining} ask_edgar call(s) remaining of "
                    f"{ASK_EDGAR_MAX_CALLS}. Prioritise the checklist items "
                    f"you have not yet covered."
                )
            if data.get("unverified"):
                out += (
                    f"\n\nWARNING — these figures in the above answer do not "
                    f"appear in any retrieved filing chunk: "
                    f"{data['unverified']}. Do not use them in the memo or in "
                    f"calculate inputs without re-retrieving them first."
                )
            return out

        if name == "extract_metrics":
            resp = await http.post(f"{API_BASE}/extract", json=inputs)
            if resp.status_code != 200:
                return f"Error from /extract_metrics: {resp.status_code} — {resp.text[:500]}"
            _record_delegated_usage(resp)
            return resp.text

        if name == "check_latest_filings":
            payload = {"ticker": inputs["ticker"]}
            # Passed through only when the agent named its forms. Omitting
            # the key lets the server auto-detect the filer's family and
            # narrow it to periodic reports; sending form_types=None would
            # be the same thing but makes the wire format depend on a null.
            if inputs.get("form_types"):
                payload["form_types"] = inputs["form_types"]
            resp = await http.post(f"{API_BASE}/latest-filings", json=payload)
            if resp.status_code != 200:
                return f"Error from /latest-filings: {resp.status_code} — {resp.text[:500]}"
            return resp.text

    return f"Unknown tool: {name}"


def _stub(name: str, inputs: dict) -> str:
    if name == "check_corpus":
        return (
            "STUB: AVGO has 3 filings, 2023-12-14 to 2025-12-18, 487 chunks."
        )
    if name == "ask_edgar":
        return (
            "STUB ANSWER: Broadcom's semiconductor solutions revenue was "
            "$36,858 million in FY2025 vs $30,096 million in FY2024, +22%. "
            "[AVGO 10-K 2025 §Item 7]"
        )
    if name == "ingest_ticker":
        return f"STUB: ingested {inputs.get('ticker')}, {inputs.get('limit', 3)} filings."
    if name == "extract_metrics":
        return "STUB: revenue=63887, gross_margin_pct=68.0, confidence=stated"
    return f"Unknown tool: {name}"

# ---------------------------------------------------------------------------
# 2. Validator — rejects literals the model didn't account for
# ---------------------------------------------------------------------------

# Scalars that belong to the operation rather than to a filing.
_OPERATION_SCALARS = {1.0, 2.0, 3.0, 4.0, 12.0, 100.0, 365.0}

# Unit-conversion multipliers — the fingerprint of a remembered figure.
_UNIT_MULTIPLIERS = {1000.0, 1000000.0, 1_000_000_000.0}


def validate_calculate_inputs(expression: str, inputs: list[dict]) -> str | None:
    """
    Five checks, in order of how cheaply they fail:
      1. no unit-conversion multipliers (fingerprint of a remembered figure)
      2. every literal in the expression is declared in `inputs`
      3. every declared input names a valid `unit` — the scale it's stated
         at in the filing. Two individually-retrieved, individually-correct
         figures (a balance-sheet line in thousands, a segment-table line
         in millions) can still combine into a wrong ratio if nothing
         normalizes them onto the same scale. Checks 1-2 and 4-5 confirm a
         figure is real and correctly attributed; none of them confirm two
         real figures were combined at the same scale — that's this check's
         job, and it's what lets safe_calculate() normalize automatically
         instead of relying on the model to convert by hand (which check 1
         already forbids doing inside the expression).
      4. every declared input actually appeared in some tool output this run
      5. every declared input's fiscal_period year actually appears near
         where that value was retrieved — catches a real value paired with
         the wrong period's label (e.g. FY2025's debt figure declared as
         FY2023's), which check 4 alone can't see: it only confirms the
         number exists somewhere in this run's retrieved text, not that it
         was retrieved *for the period being claimed*. This is the exact
         gap citation_verifier.py's docstring names as out of its scope:
         "wrong fiscal-year attribution of a real figure."
    """
    literals = {
        float(m) for m in re.findall(r"\d+\.?\d*(?:[eE][+-]?\d+)?", expression)
    }
    declared = {
        float(i["value"]) for i in inputs if i.get("value") is not None
    }
 
    # 1 — reconstruction fingerprint
    reconstructed = literals & _UNIT_MULTIPLIERS
    if reconstructed:
        return (
            f"Rejected: expression contains unit-conversion multiplier(s) "
            f"{sorted(reconstructed)}. A figure retrieved from a filing is a "
            f"single literal in the filing's own units. Retrieve the exact "
            f"value and call calculate again."
        )
 
    # 2 — undeclared literals
    unaccounted = literals - declared - _OPERATION_SCALARS
    if unaccounted:
        return (
            f"Rejected: these numbers appear in the expression but are not "
            f"declared in `inputs`: {sorted(unaccounted)}. Every figure must "
            f"be retrieved from a filing and declared with its fiscal period "
            f"and source."
        )
 
    # 3 — every declared input states a valid unit
    bad_units = [
        i for i in inputs
        if i.get("value") is not None
        and float(i["value"]) not in _OPERATION_SCALARS
        and i.get("unit") not in _UNIT_SCALES
    ]
    if bad_units:
        detail = "; ".join(
            f"{i['value']} ({i.get('label', 'unlabelled')}) unit={i.get('unit')!r}"
            for i in bad_units
        )
        return (
            f"Rejected: missing or invalid `unit` for: {detail}. Declare the "
            f"scale each figure is actually reported at in the filing — one "
            f"of {sorted(_UNIT_SCALES)}. This lets calculate normalize a "
            f"thousands-scale figure and a millions-scale figure onto the "
            f"same footing before combining them; do not convert by hand."
        )

    # 4 — declared but never returned by any tool
    corpus = _provenance_corpus()
    if corpus:
        unsourced = [
            i for i in inputs
            if i.get("value") is not None
            and float(i["value"]) not in _OPERATION_SCALARS
            and not _appears_in_output(float(i["value"]), corpus)
        ]
        if unsourced:
            detail = "; ".join(
                f"{i['value']} ({i.get('label', 'unlabelled')})"
                for i in unsourced
            )
            return (
                f"Rejected: no tool returned these figures during this run: "
                f"{detail}. You cited a source for them, but the value never "
                f"appeared in any ask_edgar or extract_metrics result. Either "
                f"retrieve the figure first, or — if you derived it yourself — "
                f"show the derivation as a separate calculate call using only "
                f"retrieved figures."
            )

        # 5 — real value, wrong period label
        outputs = _RETRIEVED_TEXT
        mismatched = []
        for i in inputs:
            if i.get("value") is None:
                continue
            value = float(i["value"])
            if value in _OPERATION_SCALARS:
                continue
            year = _fiscal_year(i.get("fiscal_period", ""))
            if year is None:
                continue  # can't check a period we can't parse a year from
            if not _year_near_any_occurrence(value, year, outputs):
                mismatched.append((i, year))

        if mismatched:
            detail = "; ".join(
                f"{i['value']} ({i.get('label', 'unlabelled')}) declared as "
                f"{i.get('fiscal_period')}, but '{year}' never appears near "
                f"any occurrence of {i['value']} in retrieved text"
                for i, year in mismatched
            )
            return (
                f"Rejected: fiscal-period mismatch — {detail}. This value "
                f"was retrieved, but not in a context mentioning the period "
                f"you're claiming it for — check whether you've paired a "
                f"figure from the wrong fiscal year (e.g. a later year's "
                f"balance used for an earlier year's ratio). Re-retrieve "
                f"the correct period's value before calling calculate again."
            )

    return None

# ---------------------------------------------------------------------------
# Run-scoped store of everything the tools have returned.
#
# Module-level, which assumes one agent run per process — true for the CLI.
# If you ever run concurrent agents in one process, make this a context
# object passed through execute_tool instead.
# ---------------------------------------------------------------------------
 
_RETRIEVED_TEXT: list[str] = []
_CALC_RESULTS: list[float] = []
_REJECTED_CALC_ATTEMPTS: list[dict] = []
_SESSION_LOG: list[str] = []

_DELEGATED_USAGE = TokenUsage()


def reset_run_provenance() -> None:
    """Call once at the start of each agent run."""
    global _DELEGATED_USAGE, _ASK_EDGAR_CALLS
    _RETRIEVED_TEXT.clear()
    _CALC_RESULTS.clear()
    _REJECTED_CALC_ATTEMPTS.clear()
    _SESSION_LOG.clear()
    _DELEGATED_USAGE = TokenUsage()
    _ASK_EDGAR_CALLS = 0
    _CALC_CACHE.clear()


def _record_delegated_usage(resp) -> None:
    """Accumulate what the API says a tool call cost.

    The agent's tools are HTTP calls to the FastAPI app, and several of them
    (`ask_edgar`, `extract_metrics`) run their own Claude calls server-side.
    Until 2026-08-27 that spend reached nothing: not the cost log, not
    `TradingState.cost_events`, and so not `check_run_guards` -- measured at
    ~28% of a real fundamentals run, enough for AVGO to exceed its $1.10 cap
    unnoticed. The server reports; the caller accounts, because only the
    caller knows the run_id.

    A missing or malformed header is treated as zero rather than an error:
    an older server, or one of the endpoints that spends nothing, should not
    break a run over accounting.
    """
    global _DELEGATED_USAGE
    raw = resp.headers.get(USAGE_HEADER)
    if not raw:
        return
    try:
        _DELEGATED_USAGE = _DELEGATED_USAGE + TokenUsage.model_validate_json(raw)
    except ValidationError:
        logger.warning("ignoring malformed %s header: %r", USAGE_HEADER, raw[:120])


def get_delegated_usage() -> TokenUsage:
    """Total server-side spend since the last `reset_run_provenance()`."""
    return _DELEGATED_USAGE

def record_log_line(text: str) -> None:
    """Append a line to the run's session log — the full terminal trace
    (tool calls, tool results, agent commentary, turn markers) saved
    beside the report for post-run auditing."""
    _SESSION_LOG.append(text)

def get_session_log() -> str:
    return "\n".join(_SESSION_LOG)

def record_calc_result(value: str | float) -> None:
    try:
        _CALC_RESULTS.append(float(value))
    except (TypeError, ValueError):
        pass

def get_calc_results() -> list[float]:
    return list(_CALC_RESULTS)

def record_rejected_calc(expression: str, inputs: list[dict], reason: str) -> None:
    """Record a calculate() call the guard rejected. Also evaluates the raw
    expression regardless of *why* it was rejected (bad unit, unsourced
    literal, fiscal-period mismatch, ...) — the arithmetic is usually still
    valid even when the guard's business-logic checks fail — so a later
    memo number that traces back to this unvalidated derivation can be
    caught even if the model never retried the call."""
    attempted = safe_calculate(expression, inputs)
    try:
        value = float(attempted)
    except (TypeError, ValueError):
        value = None
    _REJECTED_CALC_ATTEMPTS.append({
        "expression": expression,
        "reason": reason,
        "attempted_result": value,
    })

def get_unretried_rejected_calcs() -> list[dict]:
    """Rejected calculate() attempts whose would-be result was never
    matched by a later successful calculate() call in this run — i.e. the
    model used (or may use) this number without ever validating it."""
    retried = {round(v, 2) for v in _CALC_RESULTS}
    out = []
    for att in _REJECTED_CALC_ATTEMPTS:
        v = att["attempted_result"]
        if v is None or round(v, 2) in retried:
            continue
        out.append({
            "value": v,
            "reason": att["reason"],
            "expression": att["expression"],
        })
    return out

def get_provenance_corpus() -> str:
    return "\n".join(_RETRIEVED_TEXT)

def record_tool_output(text: str) -> None:
    """Record a tool result so its figures count as retrieved."""
    if text:
        _RETRIEVED_TEXT.append(text)
 
 
def _provenance_corpus() -> str:
    return "\n".join(_RETRIEVED_TEXT)

# ---------------------------------------------------------------------------
# Number matching — a tool returns "€11,384.0 million"; calculate gets 11384.0
# ---------------------------------------------------------------------------
 
def _variants(value: float) -> set[str]:
    """String forms a tool output might use for this value."""
    out: set[str] = set()
    if value == int(value):
        n = int(value)
        out.update({str(n), f"{n:,}"})
        # a tool may render a whole number with one decimal
        out.update({f"{n}.0", f"{n:,}.0"})
    else:
        out.update({
            f"{value}", f"{value:,}",
            f"{value:.1f}", f"{value:,.1f}",
            f"{value:.2f}", f"{value:,.2f}",
        })
    return out
 
 
def _appears_in_output(value: float, corpus: str) -> bool:
    return any(v in corpus for v in _variants(value))


# ---------------------------------------------------------------------------
# Fiscal-period proximity — a real value can still be paired with the wrong
# period's label. Check 4 in validate_calculate_inputs uses these.
# ---------------------------------------------------------------------------

_YEAR_IN_PERIOD_RE = re.compile(r"(19|20)\d{2}")
_PROXIMITY_WINDOW = 400  # chars either side of a value occurrence


def _fiscal_year(fiscal_period: str) -> str | None:
    """Pull the 4-digit year out of a fiscal_period label like 'FY2023',
    'Q1 2026', or 'fiscal year 2024'. None if no year is parseable —
    the caller skips the check rather than guess."""
    m = _YEAR_IN_PERIOD_RE.search(fiscal_period or "")
    return m.group() if m else None


def _occurrence_spans(value: float, text: str) -> list[tuple[int, int]]:
    """Every (start, end) span where some string form of `value` occurs."""
    spans: list[tuple[int, int]] = []
    for v in _variants(value):
        start = 0
        while True:
            idx = text.find(v, start)
            if idx == -1:
                break
            spans.append((idx, idx + len(v)))
            start = idx + len(v)
    return spans


def _year_near_any_occurrence(value: float, year: str, outputs: list[str]) -> bool:
    """True if `year` appears within _PROXIMITY_WINDOW chars of at least one
    occurrence of `value`, *within a single tool output*. A value retrieved
    for the right period is normally restated near its period somewhere in
    the same tool call's response (an ask_edgar answer's prose, or
    extract_metrics' reasoning field).

    Deliberately scoped per-output rather than to the whole run's
    concatenated corpus: two unrelated tool outputs recorded back-to-back
    can land within a few hundred chars of each other purely from being
    adjacent in the join, which would make "proximity" meaningless once
    several short outputs accumulate. Checking within one output's own text
    preserves the actual boundary of "this number and this year were
    stated together."
    """
    for text in outputs:
        for start, end in _occurrence_spans(value, text):
            window = text[max(0, start - _PROXIMITY_WINDOW): end + _PROXIMITY_WINDOW]
            if year in window:
                return True
    return False

