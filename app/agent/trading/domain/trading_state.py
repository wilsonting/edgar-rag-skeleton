import operator
from datetime import date
from typing import Annotated, TypedDict
from app.agent.trading.domain.debate import DebateTurn
from app.agent.trading.domain.decision_memo import DecisionMemo
from app.agent.trading.domain.fundamentals_report import FundamentalsReport
from app.agent.trading.domain.news_digest import NewsDigest, SentimentSummary
from app.agent.trading.domain.risk import RiskTurn
from app.agent.trading.domain.technical_report import TechnicalReport


class TradingState(TypedDict, total=False):
    ticker: str
    # Analysis date — the upper bound for ALL point-in-time data. Set once at
    # graph entry (CLI --as-of), never computed inside a node: a node calling
    # date.today() internally makes probe-date runs impossible to verify.
    as_of_date: date
    fundamentals_report: FundamentalsReport
    technical_report: TechnicalReport
    news_digest: NewsDigest
    # Structural problems the digest join flagged (missing/duplicate index,
    # invalid enum) — surfaced for review, never silently absorbed.
    news_digest_issues: list[str]
    sentiment_summary: SentimentSummary

    # The debate transcript, one entry per turn. An add-reducer, because the
    # debate is a CYCLE: bull_turn and bear_turn each run several times and a
    # plain overwrite channel would keep only the last turn. Nodes return a
    # one-element delta (`{"debate_turns": [turn]}`), never the accumulated
    # list — returning the whole list doubles it every super-step, and that
    # failure looks exactly like the runaway loop the round cap exists to
    # prevent.
    #
    # There is deliberately no separate round counter: round state IS
    # len(debate_turns). A counter beside the list is a second source of
    # truth that can desync, and a desync shows up as either an early stop or
    # a runaway — both silent.
    #
    # `operator.add` and not a dedup-on-turn_index reducer, and that was an
    # ASSUMPTION until it was tested: a task re-executing after a crash could
    # in principle append its pending write a second time, producing a
    # seven-turn transcript under a six-turn cap. VERIFIED live 2026-08-23 by
    # killing a run with os._exit(1) after the LLM call for turn 2 and
    # resuming: 6 turns, contiguous indices, turns 0 and 1 byte-identical to
    # the pre-crash snapshot. Plain add is correct here.
    debate_turns: Annotated[list[DebateTurn], operator.add]

    # Which termination layer stopped the debate: "round_cap" | "no_evidence"
    # | "". ("unproductive" existed until 2026-08-24 — removed with the
    # router clause that produced it, since it never fired live; see
    # debate_router.py.) Recorded because a capped debate reads in the memo
    # exactly like a resolved one unless the memo says otherwise.
    debate_terminated_by: str

    # The risk panel's transcript, one entry per turn. Same shape and same
    # reason as debate_turns: the panel is a CYCLE across three personas, so
    # an add-reducer is required and nodes return a one-element delta, never
    # the accumulated list. No separate round counter — round state IS
    # len(risk_turns) // len(PERSONAS), same "no second source of truth"
    # rule debate_turns follows.
    #
    # `risk_summary` (a plain str channel, one LLM-authored paragraph) is
    # DELETED, not merely renamed — Phase 6 replaced it with risk_turns plus
    # a ledger the synthesizer derives from them, the same move Phase 5 made
    # for debate_summary -> debate_turns. Rendering a summary string AND
    # keeping the structured turns would be two sources of truth for one
    # piece of content, and the one that only the synthesizer reads is the
    # one nothing would catch drifting from the other.
    risk_turns: Annotated[list[RiskTurn], operator.add]

    # Which termination layer stopped the panel: "round_cap" | "no_debate" |
    # "". Deliberately only two reachable reasons, same as debate_terminated_by
    # — there is no productivity/convergence branch to name a third one.
    risk_terminated_by: str

    decision_memo: DecisionMemo