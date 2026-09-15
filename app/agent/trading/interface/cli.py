if __name__ == "__main__":
    # Entry point: .env first, before the imports below read their settings.
    # Imported for its helpers (tests), it loads nothing. See app/config.py.
    from app.config import load_env

    load_env()

import argparse
import asyncio
import json
import logging
import sys
from datetime import date

from app.agent.researcher import _ticker_arg, vault_run
from app.agent.trading.infrastructure.checkpointer import build_checkpointer
from app.agent.trading.infrastructure.graph import ALL_ANALYSTS, build_trading_graph
from app.agent.trading.infrastructure.run_log import capture_terminal_log
# The run lifecycle lives in runner.py so the API cannot drift from it. The
# private names stay importable from here for the tests that pin them.
from app.agent.trading.interface.runner import (  # noqa: F401
    DEFAULT_MAX_USD,
    DEFAULT_WALL_CLOCK_TIMEOUT_S,
    RECURSION_LIMIT,
    _describe_stale_budget,
    _humanize,
    default_thread_id,
    save_vault_artifacts as _save_vault_artifacts,
    start_or_resume,
)


async def run(
    ticker: str,
    thread_id: str | None,
    as_of: date,
    analysts: list[str] | None,
    max_usd: float = DEFAULT_MAX_USD,
    wall_clock_timeout_s: float = DEFAULT_WALL_CLOCK_TIMEOUT_S,
) -> dict | None:
    thread_id = thread_id or default_thread_id(ticker, analysts)
    if analysts is not None:
        print(f"Analysts: {', '.join(sorted(analysts))} (others skipped)")
    async with build_checkpointer() as checkpointer:
        graph = build_trading_graph(checkpointer, analysts=analysts)
        outcome = await start_or_resume(
            graph, ticker, thread_id, as_of,
            max_usd=max_usd, wall_clock_timeout_s=wall_clock_timeout_s,
        )

    if outcome.status == "refused":
        print(outcome.refusal, file=sys.stderr)
        return None
    print({
        "completed": f"Run already completed for {ticker} (thread {thread_id})",
        "resumed": f"Resumed unfinished run for {ticker} (thread {thread_id})",
        "started": f"Started new run for {ticker} (thread {thread_id})",
    }[outcome.status])
    result = outcome.result

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
            if turn.unresolved_flags:
                print(f"    [unresolved references] {turn.unresolved_flags}")
            if turn.direction_flags:
                print(f"    [contradicted direction] {turn.direction_flags}")
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


def main() -> None:
    # This entry point configured no logging at all, so every logger.info in
    # the pipeline went nowhere and WARNING arrived only via Python's
    # handler-of-last-resort, unformatted and without a logger name. app/cli.py
    # has always done this; the trading CLI — the one that spends the most per
    # invocation, and whose failures cost a whole run — did not.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run the trading pipeline for a single ticker")
    parser.add_argument("ticker", type=_ticker_arg)
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
        # None only when a resume was refused — nothing ran, nothing to save.
        saved = _save_vault_artifacts(result, run_log()) if result is not None else []

    if saved:
        print(f"\n[vault] run {folder}: {saved[0].parent}")
    for path in saved:
        print(f"[vault]   {path.name}")


if __name__ == "__main__":
    main()