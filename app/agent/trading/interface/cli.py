import argparse
import asyncio
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone

from app.agent.researcher import vault_run
from app.agent.trading.application.debate_router import MAX_ROUNDS
from app.agent.trading.application.risk_router import RISK_MAX_ROUNDS
from app.agent.trading.domain.budget import RunBudget, RunTermination
from app.agent.trading.infrastructure.checkpointer import build_checkpointer
from app.agent.trading.infrastructure.cost_log import log_run_summary
from app.agent.trading.infrastructure.debate_port import save_debate_transcript
from app.agent.trading.infrastructure.decision_memo_port import save_decision_memo
from app.agent.trading.infrastructure.graph import (
    ALL_ANALYSTS,
    ANALYST_CHAINS,
    build_trading_graph,
)
from app.agent.trading.infrastructure.news_digest_port import save_sentiment_report
from app.agent.trading.infrastructure.risk_port import save_risk_transcript
from app.agent.trading.infrastructure.run_log import capture_terminal_log

# Gate B (Phase 6 plan §0): recursion_limit is a GLOBAL super-step budget for
# the whole invocation, not per-cycle — Phase 5's "2 * MAX_ROUNDS + 12" only
# covered one cycle plus a fixed 12 for "other nodes + slack". Phase 6 adds a
# second cycle of comparable depth (three personas instead of two sides), so
# reusing that literal would trip a legitimate run mid-risk-round with a
# GraphRecursionError that reads like a hung risk panel rather than what it
# actually is: an under-sized global counter. Every term here is derived,
# never a literal, for the same reason Phase 5's version was: a hardcoded
# number silently becomes wrong the day MAX_ROUNDS or RISK_MAX_ROUNDS moves.
_FIXED_NODES = sum(len(chain) for chain in ANALYST_CHAINS.values())  # worst case: all analysts selected
_FIXED_NODES += 4   # debate_close, risk_close, synthesizer, graceful_abort (Phase 8)
RECURSION_LIMIT = (
    2 * MAX_ROUNDS            # debate turns (bull/bear alternation)
    + 3 * RISK_MAX_ROUNDS     # risk turns (three-persona rotation)
    + _FIXED_NODES
    + 5                       # headroom, matching Phase 5's margin
)

# Phase 7 battery measured $2.24 over 5 tickers (~$0.448/run) BEFORE prompt
# caching was measured; the cache_control breakpoints already in debate_port/
# risk_port/synthesis_port were already live at that point but never
# separately verified, so 0.60/0.75 are the pre-caching-informed target/hard
# cap, to be recalibrated once a real run's cache_read_ratio is measured
# (docs/cost-log.jsonl run_summary lines, criterion 3).
DEFAULT_MAX_USD = 0.75
# No prior measurement of real end-to-end wall-clock time exists — generous
# on purpose, since the deadline exists to catch a genuine hang, not to race
# a normal run. Override with --wall-clock-timeout-s for the breach test.
DEFAULT_WALL_CLOCK_TIMEOUT_S = 1800


def _describe_stale_budget(values: dict, max_usd: float, wall_clock_timeout_s: float) -> str | None:
    """Refuse a resume whose inherited deadline has already passed, and say
    so when the inherited budget disagrees with the flags just given.

    Returns an error string to print, or None to proceed. Checked BEFORE
    `ainvoke` so a doomed resume costs nothing: the run-level guards can only
    fire between nodes, which on this graph means after the fundamentals
    stage has already been paid for.
    """
    budget = values.get("budget")
    if budget is None:
        return None

    now = datetime.now(timezone.utc)
    lines = []
    if budget.deadline_utc <= now:
        overdue = now - budget.deadline_utc
        lines.append(
            f"REFUSING TO RESUME: this thread's deadline passed "
            f"{_humanize(overdue)} ago ({budget.deadline_utc.isoformat()}).\n"
            f"The deadline is an absolute instant fixed when the run first "
            f"started, not a fresh window per attempt, so resuming would run "
            f"the expensive analyst stages and then abort on the first guard "
            f"check — paying full price for no memo."
        )
    if abs(budget.max_usd - max_usd) > 1e-9:
        lines.append(
            f"NOTE: --max-usd {max_usd:.2f} is IGNORED on a resume; this "
            f"thread carries ${budget.max_usd:.2f} from its first attempt."
        )
    if not lines:
        return None

    lines.append(
        "Start a fresh thread instead (--thread-id ...-r2), which takes the "
        "budget and deadline from this command line."
    )
    return "\n".join(lines)


def _humanize(delta: timedelta) -> str:
    hours, rem = divmod(int(delta.total_seconds()), 3600)
    return f"{hours}h{rem // 60:02d}m" if hours else f"{rem // 60}m"


async def run(
    ticker: str,
    thread_id: str | None,
    as_of: date,
    analysts: list[str] | None,
    max_usd: float = DEFAULT_MAX_USD,
    wall_clock_timeout_s: float = DEFAULT_WALL_CLOCK_TIMEOUT_S,
) -> None:
    # A subset run has a different topology, so it gets its own default thread:
    # resuming a full run's checkpoint under a narrower graph would report the
    # cached fundamentals/technical of an earlier run as if this run produced
    # them. An explicit --thread-id still overrides, deliberately.
    suffix = "" if analysts is None else "-" + "+".join(sorted(analysts))
    thread_id = thread_id or f"trading-{ticker}{suffix}"
    if analysts is not None:
        print(f"Analysts: {', '.join(sorted(analysts))} (others skipped)")
    wall_clock_start = time.monotonic()
    invoked = False  # false for an already-completed thread — see below
    async with build_checkpointer() as checkpointer:
        graph = build_trading_graph(checkpointer, analysts=analysts)
        # Layer 2 of both cycles' termination guarantee, behind each
        # router's own cap. See RECURSION_LIMIT above for the derivation.
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": RECURSION_LIMIT,
        }
        state = await graph.aget_state(config)

        if state.values and not state.next:
            print(f"Run already completed for {ticker} (thread {thread_id})")
            result = state.values
        elif state.next:
            print(f"Resuming unfinished run for {ticker} at: {state.next}")
            # A resume inherits the checkpoint's budget and deadline, never
            # the flags on THIS command line — see the comment on the
            # new-run branch below for why that rule exists. It is the right
            # rule and it had a hole: nothing said so out loud, and nothing
            # checked whether the inherited deadline was already in the past.
            #
            # Live cost of that hole (MSFT, 2026-08-28): a thread whose first
            # attempt died ~17 hours earlier was resumed with --max-usd 1.40.
            # The run silently used the checkpointed 1.10, executed the whole
            # fundamentals stage, then aborted `deadline_exceeded` on the
            # first guard check after it — $0.4069 spent, no memo. The
            # deadline is an absolute instant, so it had expired long before
            # the process started; the guard simply had no chance to say so
            # until an expensive node had already run.
            stale = _describe_stale_budget(state.values, max_usd, wall_clock_timeout_s)
            if stale:
                print(stale, file=sys.stderr)
                return None
            result = await graph.ainvoke(None, config=config)
        else:
            print(f"Starting new run for {ticker}")
            invoked = True
            # budget/run_id are set ONCE here, same rule as as_of_date — a
            # resumed run reuses whatever the checkpoint already has, never
            # recomputes deadline_utc relative to the resume time.
            budget = RunBudget(
                max_usd=max_usd,
                deadline_utc=datetime.now(timezone.utc)
                + timedelta(seconds=wall_clock_timeout_s),
            )
            result = await graph.ainvoke(
                {
                    "ticker": ticker,
                    "as_of_date": as_of,
                    "run_id": thread_id,
                    "budget": budget,
                },
                config=config,
            )

    if invoked:
        terminated_by = result.get("run_terminated_by") or RunTermination.COMPLETED
        log_run_summary(
            run_id=result.get("run_id") or thread_id,
            ticker=ticker,
            as_of_date=as_of,
            events=result.get("cost_events") or [],
            budget=result["budget"],
            terminated_by=terminated_by,
            wall_clock_s=time.monotonic() - wall_clock_start,
        )

    fundamentals = result.get("fundamentals_report")
    if fundamentals is not None:
        print("\n--- Fundamentals Report ---")
        print(fundamentals.summary)
        print(f"(tokens: in={fundamentals.input_tokens} out={fundamentals.output_tokens})")
        print("--- end Fundamentals Report ---\n")

    technical = result.get("technical_report")
    if technical is not None:
        print("\n--- Technical Report ---")
        print(f"source={technical.data_source} bars={technical.bars_used} as_of={technical.as_of_date}")
        print(technical.indicators.model_dump_json(indent=2))
        print(f"\n{technical.interpretation}")
        if technical.interpretation_flagged_numbers:
            print(f"[flagged numbers] {technical.interpretation_flagged_numbers}")
        print("--- end Technical Report ---\n")

    digest = result.get("news_digest")
    if digest is not None:
        print("\n--- News Digest ---")
        print(
            f"window={digest.window_start}..{digest.as_of_date} "
            f"items={len(digest.items)} raw={digest.raw_article_count} "
            f"truncated_by_cap={digest.truncated_by_cap}"
        )
        for item in digest.items:
            print(
                f"[{item.published_date}] {item.relevance:9} ({item.sentiment}) "
                f"{item.headline}"
            )
            print(f"    {item.summary}")
        issues = result.get("news_digest_issues") or []
        if issues:
            print(f"[digest issues] {issues}")
        print("--- end News Digest ---\n")

    sentiment = result.get("sentiment_summary")
    if sentiment is not None:
        print("\n--- Sentiment Summary ---")
        print(
            f"+{sentiment.positive} / -{sentiment.negative} / ={sentiment.neutral} "
            f"over {sentiment.article_count} articles  "
            f"net_score={sentiment.net_score:+.2f}"
        )
        if sentiment.excluded_by_relevance:
            print(
                f"({sentiment.excluded_by_relevance} of "
                f"{sentiment.article_count + sentiment.excluded_by_relevance} "
                f"digest articles excluded as not primarily about "
                f"{sentiment.ticker})"
            )
        if sentiment.article_count == 0:
            print("(no articles primarily about this company — net_score is "
                  "an absence of evidence, not neutral evidence)")
        print("--- end Sentiment Summary ---\n")

    turns = result.get("debate_turns") or []
    if turns:
        print("\n--- Bull/Bear Debate ---")
        print(
            f"{len(turns)} turn(s) over {len(turns) // 2} round(s); "
            f"terminated by {result.get('debate_terminated_by') or 'not recorded'}"
        )
        for turn in turns:
            print(
                f"\n[turn {turn.turn_index} · round {turn.round_num}] "
                f"{turn.side.upper()} stance={turn.payload.stance}"
                + (
                    f" concedes->{turn.payload.concession_trigger}"
                    if turn.payload.concession_trigger
                    else ""
                )
                + ("" if turn.productive else " (unproductive)")
            )
            print(f"    {turn.payload.argument}")
            for claim in turn.payload.claims:
                print(f"    · {claim.claim_id} [{claim.evidence_ref}] {claim.text}")
            if turn.guard_flags:
                print(f"    [flagged numbers] {turn.guard_flags}")
            if turn.unquoted_evidence:
                print(f"    [unverified quotes] {turn.unquoted_evidence}")
        total = sum(t.estimated_cost_usd or 0.0 for t in turns)
        print(f"\ndebate cost: ${total:.4f}")
        print("--- end Bull/Bear Debate ---\n")
    elif result.get("debate_terminated_by"):
        print(
            f"\n[debate] skipped: {result['debate_terminated_by']} — this run "
            f"carries no adversarial review of its analyst findings\n"
        )

    risk_turns = result.get("risk_turns") or []
    if risk_turns:
        print("\n--- Risk Panel ---")
        print(
            f"{len(risk_turns)} turn(s) over {len(risk_turns) // 3} round(s); "
            f"terminated by {result.get('risk_terminated_by') or 'not recorded'}"
        )
        for turn in risk_turns:
            print(f"\n[turn {turn.turn_index} · round {turn.round_num}] {turn.persona.upper()}")
            print(f"    {turn.payload.argument}")
            for factor in turn.payload.proposes:
                print(f"    + {factor.factor_id} {factor.text} (trigger: {factor.trigger})")
            for score in turn.payload.scores:
                print(f"    · {score.factor_id} severity={score.severity} likelihood={score.likelihood}")
            if turn.guard_flags:
                print(f"    [flags] {turn.guard_flags}")
        total = sum(t.estimated_cost_usd or 0.0 for t in risk_turns)
        print(f"\nrisk panel cost: ${total:.4f}")
        print("--- end Risk Panel ---\n")
    elif result.get("risk_terminated_by"):
        print(
            f"\n[risk] skipped: {result['risk_terminated_by']} — this run "
            f"carries no risk-panel review\n"
        )

    memo = result.get("decision_memo")
    if memo is not None:
        print(json.dumps(memo.model_dump(mode="json"), indent=2))
    elif result.get("run_terminated_by"):
        print(
            f"\n[abort] run terminated by {result['run_terminated_by'].value} "
            f"before a memo was produced — see the vault's decision-ABORTED "
            f"artifact for what the run did reach.\n"
        )
    return result


def _save_vault_artifacts(result: dict, run_log: str) -> list:
    """Write the run's artifacts once the terminal log is complete.

    Saved here rather than inside the nodes for two reasons: the log is only
    whole at the end of the run, and a resumed or already-completed run
    replays state without executing any node, which would otherwise write no
    artifact at all for a run the user just asked for.
    """
    saved = []
    digest = result.get("news_digest")
    sentiment = result.get("sentiment_summary")
    has_sentiment = digest is not None and sentiment is not None

    if has_sentiment:
        saved.append(
            save_sentiment_report(
                digest,
                sentiment,
                issues=result.get("news_digest_issues") or [],
                provenance=run_log,
            )
        )

    turns = result.get("debate_turns") or []
    if turns:
        saved.append(
            save_debate_transcript(
                result["ticker"],
                turns,
                result.get("debate_terminated_by") or "",
            )
        )

    risk_turns = result.get("risk_turns") or []
    if risk_turns:
        saved.append(
            save_risk_transcript(
                result["ticker"],
                risk_turns,
                result.get("risk_terminated_by") or "",
            )
        )

    memo = result.get("decision_memo")
    if memo is not None:
        # The log is written exactly once per run. It rides with the
        # sentiment report when there is one, and falls back to the memo
        # otherwise (e.g. `--only technical`) so a run never loses its trace.
        # Both land in the same dated folder, so the log is beside either.
        saved.append(
            save_decision_memo(memo, provenance=None if has_sentiment else run_log)
        )
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the trading pipeline for a single ticker")
    parser.add_argument("ticker")
    parser.add_argument("--thread-id", default=None)
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=date.today(),  # today() appears exactly once, at the boundary
        help="Analysis date. All news is bounded at or before this date.",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=ALL_ANALYSTS,
        metavar="ANALYST",
        help=(
            "Run only this analyst; repeat to select several "
            f"(choices: {', '.join(ALL_ANALYSTS)}). Default: all of them. "
            "The synthesizer still runs and records the others as data gaps."
        ),
    )
    parser.add_argument(
        "--max-usd",
        type=float,
        default=DEFAULT_MAX_USD,
        help=f"Hard per-run cost cap. Default: ${DEFAULT_MAX_USD:.2f}.",
    )
    parser.add_argument(
        "--wall-clock-timeout-s",
        type=float,
        default=DEFAULT_WALL_CLOCK_TIMEOUT_S,
        help=f"Hard per-run wall-clock deadline, in seconds. Default: {DEFAULT_WALL_CLOCK_TIMEOUT_S}.",
    )
    args = parser.parse_args()

    # The capture wraps the whole run so the provenance file holds the real
    # terminal session — node progress on stdout and the research agent's
    # traces on stderr, interleaved in the order they actually happened.
    # The "saved to" lines below are printed after the log is read, so they
    # are the only run output the file does not contain.
    # The run folder is opened around the WHOLE run, not around the vault
    # writes at the end. technical and fundamentals save from inside their
    # nodes while the graph is still executing; sentiment, decision and the
    # debate transcript save here afterwards. Only a folder fixed before the
    # first of those puts them all in one place.
    with capture_terminal_log() as run_log, vault_run() as folder:
        result = asyncio.run(
            run(
                args.ticker, args.thread_id, args.as_of, args.only,
                max_usd=args.max_usd,
                wall_clock_timeout_s=args.wall_clock_timeout_s,
            )
        )
        saved = _save_vault_artifacts(result, run_log())

    if saved:
        print(f"\n[vault] run {folder}: {saved[0].parent}")
    for path in saved:
        print(f"[vault]   {path.name}")


if __name__ == "__main__":
    main()