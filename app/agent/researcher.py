
"""
Research agent — orchestration loop.
 
Runs a tool-use conversation: the model plans, calls tools, reads results,
and continues until it produces a final answer or hits the turn limit.
 
Usage:
    # Step-1 loop verification (stubbed tools, test prompt):
    uv run python -m app.agent.researcher --test
 
    # Full research run:
    uv run python -m app.agent.researcher AVGO
 
    # News assessment against watchlist thesis:
    uv run python -m app.agent.researcher --news AVGO "Broadcom announces 10B share repurchase"
"""

from __future__ import annotations  
import argparse
import asyncio
import contextlib
import json
import logging
import os
from pathlib import Path
import sys
import yaml

from dotenv import load_dotenv
from datetime import datetime
from app.agent.prompts import ANALYST_SYSTEM_PROMPT, STEP1_TEST_PROMPT, NEWS_ASSESSMENT_PROMPT
from app.agent.tools import TOOLS, execute_tool, get_calc_results, get_provenance_corpus, get_session_log, get_unretried_rejected_calcs, record_log_line, reset_run_provenance
from app.application.memo_verifier import verify_memo
from app.infrastructure.llm import MODEL_PRICING, get_client
from app.infrastructure.llm.models import model_for

load_dotenv()
logger = logging.getLogger(__name__)

AGENT_MODEL = model_for("agent")
MAX_TURNS = int(os.environ["LOOP_MAX_TURNS"])
# How many turns out from the cap the agent starts being told to wrap up.
# Phase 9 measured 2 of 3 fundamentals runs hitting MAX_TURNS exactly and
# ending on "forcing memo from gathered data" — the agent had no idea the
# budget existed, so it was writing its memo under a guillotine rather than
# to a deadline. 8 is roughly what the final memo turn plus a couple of
# closing retrievals need.
TURN_WARN_AT = 8
# The 12-item memo with citations and tables routinely runs 8-10k output
# tokens on its own, before any tool-call turns. The previous 4096 cap was
# well under that, so the final memo-writing turn was silently cut off
# mid-sentence — Haiku 4.5's actual ceiling is 64000 (verified via
# client.models.retrieve), so 16000 leaves ample headroom without risking
# the non-streaming client's read timeout.
AGENT_MAX_TOKENS = 16000
WATCHLIST_PATH = Path("watchlist.yaml")
MEMO_DIR = Path.home() / os.environ["MEMO_DIR"]

# Cost config — per million tokens. The table moved to
# app/infrastructure/llm/pricing.py when the provider layer landed, because
# pricing is a property of the provider rather than of this agent. Re-exported
# under the old name so the three ports and the budget assertions that import
# `_MODEL_PRICING` from here keep working.
_MODEL_PRICING = MODEL_PRICING


def _trace(msg: str) -> None:
    """Print to stderr so tool traces don't pollute the memo output.
    Also recorded in the session log saved beside the report."""
    print(msg, file=sys.stderr)
    record_log_line(msg)

def _load_watchlist() -> list[dict]:
    """Load the watchlist YAML. Returns empty list if missing."""
    if not WATCHLIST_PATH.exists():
        _trace(f"Warning: {WATCHLIST_PATH} not found")
        return []
    return yaml.safe_load(WATCHLIST_PATH.read_text()) or []

def _get_watchlist_entry(ticker: str) -> dict | None:
    """Find the watchlist entry for a ticker. Case-insensitive."""
    watchlist = _load_watchlist()
    ticker_upper = ticker.upper()
    for entry in watchlist:
        if entry.get("ticker", "").upper() == ticker_upper:
            return entry
    return None

def _build_news_prompt(ticker: str, news_text: str) -> str:
    """
    Build the news assessment prompt with the watchlist context filled in.
    If the ticker isn't in the watchlist, uses a generic framing.
    """
    entry = _get_watchlist_entry(ticker)
 
    if entry:
        thesis = entry.get("thesis", "No thesis specified.")
        key_metrics = "\n".join(
            f"- {m}" for m in entry.get("key_metrics", [])
        ) or "- None specified"
        risks_watching = "\n".join(
            f"- {r}" for r in entry.get("risks_watching", [])
        ) or "- None specified"
    else:
        _trace(f"Warning: {ticker} not in watchlist — using generic framing")
        thesis = "No thesis on file. Assess the news on its own merits."
        key_metrics = "- None specified (ticker not in watchlist)"
        risks_watching = "- None specified (ticker not in watchlist)"
 
    prompt = NEWS_ASSESSMENT_PROMPT.format(
        ticker=ticker.upper(),
        thesis=thesis,
        key_metrics=key_metrics,
        risks_watching=risks_watching,
    )
 
    return prompt

# Modes whose provenance must NOT fall back to the research agent's session
# log. These are trading-pipeline artifacts that never call the research
# tools, so at their save time the module-global log still holds whatever
# trace the preceding fundamentals run left behind — writing it would pair
# the wrong evidence with the report. They supply their own provenance or
# get none.
_NO_SESSION_LOG_MODES = {
    "technical", "sentiment", "decision", "decision_failed", "decision_aborted",
    "debate", "risk",
}


# The vault filename stem for each mode, minus the ticker. "fundamentals" is
# the mode but "fundamental" is the filename, which is a wart old enough to
# be in people's muscle memory — kept rather than fixed, since renaming it
# would orphan every existing note's links.
_MODE_STEMS = {
    "news": "news",
    "technical": "technical",
    "fundamentals": "fundamental",
    "sentiment": "sentiment",
    "decision": "decision",
    "decision_failed": "decision-FAILED",
    "decision_aborted": "decision-ABORTED",
    "debate": "debate",
    "risk": "risk",
}

# Modes that file under MEMO_DIR/<ticker>/<date>/ rather than flat under the
# ticker. Everything the trading pipeline writes, which is what makes a
# per-run folder worth having at all.
_DATED_MODES = frozenset(
    {
        "technical", "fundamentals", "sentiment", "decision", "decision_failed",
        "decision_aborted", "debate", "risk",
    }
)

# The instant one pipeline run started, set by `vault_run`; None outside one.
#
# A module global rather than a parameter threaded through six ports, because
# the two halves of a run save at different times and through different call
# stacks — technical and fundamentals from inside their nodes while the graph
# is still executing, sentiment/decision/debate from the CLI after it
# finishes. Anything computed per call (including datetime.now()) puts those
# halves in different folders, which is the problem being fixed.
#
# The whole datetime, not the formatted name: the DATE folder has to come
# from the same instant too. A run that starts at 23:58 and finishes at
# 00:02 would otherwise file its fundamentals under one date and its debate
# transcript under the next — the same scattering, harder to spot.
_RUN_STAMP: datetime | None = None


@contextlib.contextmanager
def vault_run(stamp: datetime | None = None):
    """Give every artifact saved inside this block ONE run folder.

    Yields the folder name. The directory is not created here — it is created
    by the first `_save_output` that lands in it, so a run that dies before
    writing anything leaves no empty folder behind.

    Restores whatever was set before rather than clearing to None, so nesting
    is safe even though nothing nests today.
    """
    global _RUN_STAMP
    previous = _RUN_STAMP
    _RUN_STAMP = stamp or datetime.now()
    try:
        yield _RUN_STAMP.strftime(_RUN_FOLDER_FORMAT)
    finally:
        _RUN_STAMP = previous


_RUN_FOLDER_FORMAT = "%Y-%m%d-%H%M%S"


def _save_output(
    content: str,
    ticker: str,
    mode: str,
    cost_usd: float | None = None,
    provenance: str | None = None,
    model: str = AGENT_MODEL,
) -> Path:
    """Save output with timestamp. Returns the path.

    `provenance`, when given, is written verbatim to the sidecar file instead
    of the research agent's session log.

    Inside a `vault_run` block the artifacts of one run share a folder and
    drop the per-file timestamp:

        <ticker>/20260822/2026-0822-070153/ACN-fundamental.md

    Outside one — the standalone research CLI, which writes a single report —
    the old flat layout is kept, timestamp in the filename.
    """
    if cost_usd is not None:
        content = content.rstrip("\n") + f"\n\n---\n**LLM cost:** ${cost_usd:.4f} ({model})\n"
    # Inside a run, every path is derived from the instant the RUN started,
    # not the instant this file happens to be written.
    now = _RUN_STAMP or datetime.now()
    stem = _MODE_STEMS.get(mode)
    parent = MEMO_DIR / ticker
    if mode in _DATED_MODES:
        parent = parent / now.strftime("%Y%m%d")

    if _RUN_STAMP is not None:
        # One folder per run, so the timestamp is on the folder and not
        # repeated on every file inside it.
        parent = parent / now.strftime(_RUN_FOLDER_FORMAT)
        filename = f"{ticker}-{stem}.md" if stem else f"{ticker}.md"
    else:
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        filename = (
            f"{ticker}-{stem}-{timestamp}.md" if stem else f"{ticker}-{timestamp}.md"
        )

    out_path = parent / filename
    if _RUN_STAMP is not None and out_path.exists():
        # Two artifacts of the same kind in one run. Unreachable today —
        # every mode is saved exactly once per run — so if it happens the
        # honest answer is to say so rather than overwrite a report that
        # cost real money to produce.
        raise FileExistsError(
            f"{out_path} already exists in this run's folder — a second "
            f"'{mode}' artifact would overwrite the first. Nothing was written."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)

    # Audit artifact: the run's full session log — every terminal trace
    # line plus untruncated tool results — saved beside the report so any
    # figure in it can be traced to (or shown absent from) the exact turn
    # and tool output that produced it, instead of reconstructing the run
    # by inference after the process exits. Saved as .md (not .txt) so
    # Obsidian's file explorer, which hides unknown extensions by default,
    # shows it beside its report. See _NO_SESSION_LOG_MODES for which modes
    # must not inherit this log.
    if provenance is None and mode not in _NO_SESSION_LOG_MODES:
        provenance = get_session_log()
    if provenance and provenance.strip():
        out_path.with_name(f"{out_path.stem}-provenance.md").write_text(provenance)

    return out_path

def _print_usage_summary(
    total_input: int,
    total_cache_write: int,
    total_cache_read: int,
    total_output: int,
) -> None:
    """Print token usage and estimated cost to stderr."""
    pricing = _MODEL_PRICING.get(AGENT_MODEL)
    _trace(f"\n{'='*55}")
    _trace("[usage summary]")
    _trace(f"  model: {AGENT_MODEL}")
    if not pricing:
        _trace(
            f"  input={total_input:,}  cache_write={total_cache_write:,}  "
            f"cache_read={total_cache_read:,}  output={total_output:,}"
        )
        _trace(f"  (pricing not configured for this model)")
        _trace(f"{'='*55}")
        return

    cost_input = total_input * pricing["input"] / 1_000_000
    cost_cache_write = total_cache_write * pricing["cache_write"] / 1_000_000
    cost_cache_read = total_cache_read * pricing["cache_read"] / 1_000_000
    cost_output = total_output * pricing["output"] / 1_000_000
    total_cost = cost_input + cost_cache_write + cost_cache_read + cost_output

    _trace(f"  input:       {total_input:>9,} tokens  ${cost_input:.4f}")
    _trace(f"  cache_write: {total_cache_write:>9,} tokens  ${cost_cache_write:.4f}")
    _trace(f"  cache_read:  {total_cache_read:>9,} tokens  ${cost_cache_read:.4f}")
    _trace(f"  output:      {total_output:>9,} tokens  ${cost_output:.4f}")
    _trace(f"  ─────────────────────────────────")
    _trace(f"  TOTAL COST:  ${total_cost:.4f}")
    _trace(f"{'='*55}")


def _compute_cost(usage: UsageSummary, model: str = AGENT_MODEL) -> float | None:
    """Estimate USD cost from token usage, or None if pricing isn't configured
    for `model`.

    `model` defaults to AGENT_MODEL, which was correct by coincidence until
    Phase 5: every earlier caller used the researcher's own model. A node
    that calls a different model and does not pass it here gets a cost priced
    at the wrong rate — understated 3x for Sonnet against Haiku — which is
    exactly the kind of wrong number a per-run budget assertion would then
    wave through."""
    pricing = _MODEL_PRICING.get(model)
    if not pricing:
        return None
    return round(
        usage.input_tokens * pricing["input"] / 1_000_000
        + usage.cache_write_tokens * pricing["cache_write"] / 1_000_000
        + usage.cache_read_tokens * pricing["cache_read"] / 1_000_000
        + usage.output_tokens * pricing["output"] / 1_000_000,
        6,
    )


def log_cost(
    ticker: str,
    mode: str,
    usage: UsageSummary,
    model: str = AGENT_MODEL,
    *,
    run_id: str | None = None,
    event_id: str | None = None,
) -> float | None:
    """Append one JSON line to docs/cost-log.jsonl. Returns the estimated
    cost (or None if pricing isn't configured), so callers can also surface
    it elsewhere (e.g. in the memo itself).

    Pass `model` whenever the call being logged did not use AGENT_MODEL. Both
    the logged label and the price come from it — a hardcoded label on a
    differently-priced call produces a log that is wrong twice and looks
    right.

    `run_id`/`event_id` (Phase 8) are None for calls outside the trading
    graph — the standalone researcher CLI has no run to group lines under.
    Trading-pipeline callers pass both (see
    trading/infrastructure/cost_log.py) so a run's lines can be grouped and
    a resumed run's duplicate write can be deduped, both by `jq` over this
    file alone. `kind` distinguishes these lines from the pre-Phase-8 lines
    already in this file, which have neither field and are read as legacy."""
    cost = _compute_cost(usage, model)
    entry = {
        "kind": "cost_event",
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "event_id": event_id,
        "ticker": ticker,
        "mode": mode,
        "model": model,
        "input_tokens": usage.input_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "output_tokens": usage.output_tokens,
        "estimated_cost_usd": cost,
    }
    log_path = Path("docs/cost-log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return cost


def _roll_cache_breakpoint(messages: list) -> None:
    """Place a single cache_control breakpoint on the last block of the last
    turn, and strip any stale ones. The cached prefix is tools + system + all
    prior messages, so each turn reads the whole growing history from cache
    (~0.1x) instead of re-billing it at full price."""
    for msg in messages:
        content = msg["content"]
        if isinstance(content, list):
            for block in content:
                # response.content blocks are SDK objects, not dicts — skip them
                if isinstance(block, dict):
                    block.pop("cache_control", None)

    last = messages[-1]["content"]
    if isinstance(last, str):
        # Wrap a bare string turn so we can attach the breakpoint.
        messages[-1]["content"] = last = [{"type": "text", "text": last}]
    if isinstance(last[-1], dict):
        last[-1]["cache_control"] = {"type": "ephemeral"}



class UsageSummary:
    __slots__ = ("input_tokens", "cache_write_tokens", "cache_read_tokens", "output_tokens")

    def __init__(self) -> None:
        self.input_tokens = 0
        self.cache_write_tokens = 0
        self.cache_read_tokens = 0
        self.output_tokens = 0


def _strip_preamble(text: str) -> str:
    """Drop any scratch/transition text the model wrote before the output's
    top-level heading (e.g. "Let me compile the research memo.")."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("# "):
            return "\n".join(lines[i:])
    return text


async def run_agent(user_task: str, system_prompt: str) -> tuple[str, UsageSummary]:
    """
    Run the agent loop: send task, process tool calls, return final text
    and accumulated token usage.
    Tool traces go to stderr; only the final output goes to stdout.
    """
    reset_run_provenance()
    client = get_client(AGENT_MODEL)
    # The budget goes in the TASK, not the system prompt. The system block
    # carries its own cache breakpoint and is identical across every run;
    # interpolating a runtime number into it would rewrite that cache
    # whenever the config moved, for a line that belongs with the task
    # anyway. Stated up front rather than only warned about near the end,
    # so the agent can plan the checklist against it — the same reason
    # ask_edgar's cap is in its tool description.
    messages = [{
        "role": "user",
        "content": (
            f"{user_task}\n\n"
            f"You have {MAX_TURNS} tool-calling turns for this entire "
            f"analysis, and writing the memo takes several of them. Budget "
            f"accordingly: cover the checklist broadly before going deep on "
            f"any one item, and do not leave the memo to the last turn."
        ),
    }]
    usage = UsageSummary()

    for turn in range(MAX_TURNS):
        _trace(f"\n--- turn {turn + 1} ---")
        _roll_cache_breakpoint(messages)
        response = await client.messages.create(
            model=AGENT_MODEL,
            max_tokens=AGENT_MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=TOOLS,
            messages=messages,
        )
        u = response.usage
        usage.input_tokens += u.input_tokens
        usage.cache_write_tokens += u.cache_creation_input_tokens
        usage.cache_read_tokens += u.cache_read_input_tokens
        usage.output_tokens += u.output_tokens
        _trace(
            f"  [tokens] in={u.input_tokens} "
            f"cache_write={u.cache_creation_input_tokens} "
            f"cache_read={u.cache_read_input_tokens} out={u.output_tokens}"
        )

        for block in response.content:
            if block.type == "text" and block.text.strip():
                _trace(f"  [agent] {block.text.strip()}")

        if response.stop_reason != "tool_use":
            final = "".join(
                b.text for b in response.content if b.type == "text"
            )

            if response.stop_reason == "max_tokens":
                # The model was hard-cut by the output-length limit while
                # writing prose (not mid-tool-call) — AGENT_MAX_TOKENS should
                # make this rare, but a dense enough memo can still exceed
                # it. Give it one continuation call rather than silently
                # returning a memo that stops mid-sentence.
                _trace("  [truncated] hit max_tokens while writing the memo — requesting continuation")
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was cut off by the output "
                        "length limit before the memo was finished. Continue "
                        "writing EXACTLY where you left off — do not repeat "
                        "any text already written, do not restart the memo "
                        "— and make sure you reach a complete Assessment "
                        "section."
                    ),
                })
                cont = await client.messages.create(
                    model=AGENT_MODEL,
                    max_tokens=AGENT_MAX_TOKENS,
                    system=[
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=messages,
                )
                cu = cont.usage
                usage.input_tokens += cu.input_tokens
                usage.cache_write_tokens += cu.cache_creation_input_tokens
                usage.cache_read_tokens += cu.cache_read_input_tokens
                usage.output_tokens += cu.output_tokens
                final += "".join(b.text for b in cont.content if b.type == "text")

                final = _strip_preamble(final)
                if cont.stop_reason == "max_tokens" or "## Assessment" not in final:
                    final = (
                        "**INCOMPLETE — this memo was cut off before all "
                        "sections were completed.**\n\n"
                    ) + final
            else:
                final = _strip_preamble(final)

            final = verify_memo(final, get_provenance_corpus(), get_calc_results(),
                               get_unretried_rejected_calcs())
            _trace(f"\n[agent finished after {turn + 1} turns]")
            return final, usage

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = await execute_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        messages.append({"role": "assistant", "content": response.content})
        # Tool-result blocks must come first in a user message; a trailing
        # text block is legal after them.
        content = list(tool_results)
        remaining = MAX_TURNS - (turn + 1)
        if 0 < remaining <= TURN_WARN_AT:
            # Only inside the warn band, for the same reason the ask_edgar
            # counter is: every one of these lines lands in the conversation
            # the agent re-reads on every subsequent turn, and a countdown
            # appended to all 45 turns would be 45 copies of a changing
            # number in the context the containment guards scan.
            content.append({
                "type": "text",
                "text": (
                    f"[BUDGET] {remaining} turn(s) remaining before the memo "
                    f"is forced. Stop opening new lines of enquiry. Finish "
                    f"the checklist item you are on, then write the memo, "
                    f"recording anything you could not cover under Data Gaps."
                ),
            })
        messages.append({"role": "user", "content": content})

    # Budget exhausted — force a memo from whatever was gathered.
    _trace("\n[MAX_TURNS reached — forcing memo from gathered data]")
    messages.append({
        "role": "user",
        "content": (
            "You have exhausted your tool-call budget. Write the memo now "
            "using only data you have already retrieved. For any checklist "
            "item you could not complete, list it under Data Gaps and note "
            "that the tool budget was exhausted. Do not call any more tools."
        ),
    })
    response = await client.messages.create(
        model=AGENT_MODEL,
        max_tokens=AGENT_MAX_TOKENS,
        system=system_prompt,
        messages=messages,
    )
    u = response.usage
    usage.input_tokens += u.input_tokens
    usage.cache_write_tokens += u.cache_creation_input_tokens
    usage.cache_read_tokens += u.cache_read_input_tokens
    usage.output_tokens += u.output_tokens

    final = "".join(b.text for b in response.content if b.type == "text")
    final = _strip_preamble(final)
    if response.stop_reason == "max_tokens" or "## Assessment" not in final:
        final = (
            "**INCOMPLETE — this memo was cut off before all sections were "
            "completed.**\n\n"
        ) + final
    final = verify_memo(final, get_provenance_corpus(), get_calc_results(),
                        get_unretried_rejected_calcs())
    return final, usage


def main() -> None:
    parser = argparse.ArgumentParser(description="EDGAR research agent")
    parser.add_argument(
        "ticker", nargs="?", help="Ticker to research (omit with --test)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the step-1 loop verification against stubbed tools",
    )
    parser.add_argument(
        "--news",
        type=str,
        metavar="HEADLINE",
        help='News headline or announcement to assess, e.g. --news "AVGO announces 10B buyback"',
    )
    args = parser.parse_args()
    today = datetime.today().isoformat()
    task = f"Today's date is {today}. "
    if args.test:
        task += "Run the test task described in your instructions."
        prompt = STEP1_TEST_PROMPT
    elif args.ticker and args.news:
        ticker = args.ticker.upper()
        prompt = _build_news_prompt(ticker, args.news)
        task += f"Assess this news for {ticker}:\n\n{args.news}"
        mode = "news"
    elif args.ticker:
        task += f"Run the full research checklist for {args.ticker}."
        prompt = ANALYST_SYSTEM_PROMPT
        mode = "research"
    else:
        parser.error("provide a ticker, or use --test")

    result, usage = asyncio.run(run_agent(task, prompt))
    print(result)

    cost = log_cost(args.ticker.upper(), mode, usage) if args.ticker else None

    # Save to file (skip for test mode)
    if mode != "test" and args.ticker:
        path = _save_output(result, args.ticker.upper(), mode, cost_usd=cost)
        _trace(f"\nSaved to {path}")

    _print_usage_summary(
        usage.input_tokens, usage.cache_write_tokens,
        usage.cache_read_tokens, usage.output_tokens,
    )


if __name__ == "__main__":
    main()