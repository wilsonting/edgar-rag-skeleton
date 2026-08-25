from __future__ import annotations

import re
from pathlib import Path

from anthropic import AsyncAnthropic

from app.agent.researcher import AGENT_MODEL, UsageSummary, _save_output, log_cost
from app.agent.trading.domain.technical_report import TechnicalIndicators, TechnicalReport

TECHNICAL_INTERPRETER_SYSTEM_PROMPT = """\
You are a technical analysis interpreter. You will be given a set of already-computed
indicator values for a stock. Your job is ONLY to interpret these values in plain
language — trend direction, momentum, overbought/oversold condition, volatility regime,
and volume context.

STRICT RULE: Do not calculate, recompute, restate with different precision, or invent
ANY numeric value. Every number in your response must be one of the numbers given to
you, used exactly as given (you may round for readability, e.g. 62.37 -> "around 62").
If you are not given a value (None), do not guess or fabricate one — say the signal
is unavailable.

USE THE GIVEN RELATIONS: you will be shown a "Computed relations" block stating how
the values compare to each other — whether price is above or below each moving
average, how the moving averages sit relative to one another, and so on. Those
comparisons are computed in code and are authoritative. State them as given. Do not
work out any comparison yourself from the raw numbers, and never contradict the
block. In particular, where price sits relative to a moving average and where the
moving averages sit relative to each other are two different facts — do not
substitute one for the other.

MACD PRECISION: `macd`, `macd_signal`, and `macd_histogram` are three distinct values —
never refer to any of them as just "the MACD". A negative `macd_histogram` means the
MACD line is below its signal line (a bearish crossover), NOT that the MACD line
itself is below zero — those are different conditions and must not be conflated. Name
the specific line you mean: "the MACD line", "the signal line", or "the histogram".

Respond in 3-5 sentences of plain-language interpretation. No preamble, no headers.
"""


_MA_LABELS = {"sma_50": "50-day average", "sma_200": "200-day average"}


def derive_relations(ind: TechnicalIndicators) -> list[str]:
    """State the comparisons in code rather than leaving them to the model.

    A live MSFT run produced "trading above its 50-day moving average (around
    419) but below its 200-day average (around 429)" while the last close was
    483.24 — above both. The model had collapsed two different facts, where
    price sits relative to each average and where the averages sit relative to
    each other, into one wrong claim. Every number in that sentence was
    genuine, so the flagged-numbers guard had nothing to catch.

    This is the same move as joining news items by index instead of letting
    the model retype them: whatever Python can decide, Python decides.
    """
    close = ind.last_close
    rel: list[str] = []

    for field, label in _MA_LABELS.items():
        value = getattr(ind, field)
        if value is not None:
            side = "ABOVE" if close > value else "BELOW"
            rel.append(f"last close ({close:.2f}) is {side} the {label} ({value:.2f})")

    if ind.sma_50 is not None and ind.sma_200 is not None:
        side = "ABOVE" if ind.sma_50 > ind.sma_200 else "BELOW"
        rel.append(
            f"the 50-day average ({ind.sma_50:.2f}) is {side} the 200-day average "
            f"({ind.sma_200:.2f}) — this is a statement about the two averages, "
            f"NOT about where price sits"
        )

    if ind.macd is not None and ind.macd_signal is not None:
        side = "ABOVE" if ind.macd > ind.macd_signal else "BELOW"
        rel.append(f"the MACD line ({ind.macd:.4f}) is {side} its signal line "
                   f"({ind.macd_signal:.4f})")

    if ind.rsi_14 is not None:
        band = (
            "OVERBOUGHT (>70)" if ind.rsi_14 > 70
            else "OVERSOLD (<30)" if ind.rsi_14 < 30
            else "NEITHER overbought nor oversold (between 30 and 70)"
        )
        rel.append(f"RSI ({ind.rsi_14:.2f}) is {band}")

    if ind.bb_upper is not None and ind.bb_lower is not None:
        where = (
            "ABOVE the upper band" if close > ind.bb_upper
            else "BELOW the lower band" if close < ind.bb_lower
            else "WITHIN the bands"
        )
        rel.append(f"last close ({close:.2f}) is {where} "
                   f"({ind.bb_lower:.2f} to {ind.bb_upper:.2f})")

    if ind.volume_vs_20d_avg is not None:
        side = "ABOVE" if ind.volume_vs_20d_avg > 1 else "BELOW"
        rel.append(f"latest volume is {side} its 20-day average "
                   f"({ind.volume_vs_20d_avg:.4f}x)")

    return rel


async def interpret_indicators(
    ticker: str, indicators: TechnicalIndicators
) -> tuple[str, list[str], list[str], float | None]:
    client = AsyncAnthropic()
    relations = "\n".join(f"- {r}" for r in derive_relations(indicators))
    prompt = (
        f"Ticker: {ticker}\n"
        f"Indicators:\n{indicators.model_dump_json(indent=2)}\n\n"
        f"Computed relations (authoritative — state these as given, do not\n"
        f"re-derive them from the numbers above):\n{relations}\n\n"
        "Provide the interpretation now."
    )
    response = await client.messages.create(
        model=AGENT_MODEL,
        max_tokens=512,
        system=TECHNICAL_INTERPRETER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    interpretation = "".join(b.text for b in response.content if b.type == "text")

    usage = UsageSummary()
    u = response.usage
    usage.input_tokens = u.input_tokens
    usage.cache_write_tokens = u.cache_creation_input_tokens
    usage.cache_read_tokens = u.cache_read_input_tokens
    usage.output_tokens = u.output_tokens
    cost = log_cost(ticker, "trading-technical", usage)

    flagged = _flag_unmatched_numbers(interpretation, indicators)
    flagged_claims = flag_contradicted_claims(interpretation, indicators)
    return interpretation, flagged, flagged_claims, cost


# A claim that something is above/below an N-day average. Non-greedy up to the
# period so "above its 50-day moving average" and "below the 200-day" both
# match, but the search never runs past a sentence boundary.
_PRICE_VS_MA = re.compile(r"\b(above|below)\b[^.;]{0,40}?\b(\d+)-day", re.I)

# The same words describe a different claim when an average is the subject:
# "the 50-day moving average is below the 200-day" compares two averages, and
# checking it against price would flag correct prose. The exclusion has to be
# tight — requiring the average-plus-verb to sit immediately against the
# comparator — because the real failure read "...moving average (around 419)
# but below its 200-day average", where an average appears shortly before
# `below` and yet price is still the subject. A looser rule would have
# skipped exactly the sentence this guard exists to catch.
# Both halves of this pattern were forced by live output, not designed up
# front, and each round of tightening is worth keeping in view:
#
#   1. Enumerating verbs failed. The model wrote "the 50-day average SITTING
#      below the 200-day average" — a participle, not the finite "sits" — and
#      correct prose was reported as a contradiction. Hence up to two bare
#      words instead of a verb list.
#   2. Requiring the noun failed. The next run wrote "the 50-day SITTING below
#      the 200-day", eliding "average" entirely. Hence the noun is optional.
#
# What still separates the two claims is what sits between the average and
# the comparator. In the real error it was "(around 419) but " — parentheses
# and a digit, not bare words — so the exclusion does not fire and the false
# claim is caught. Two words is the whole margin; widening it would start
# swallowing "...50-day average but the price is below its 200-day".
#
# A guard that cries wolf is worse than no guard: it teaches the reader to
# skip it, and then it is not there for the one that matters.
_MA_IS_SUBJECT = re.compile(
    r"\d+-day(?:\s+(?:simple|moving|exponential))*(?:\s+(?:average|ma|sma)s?)?\s+"
    r"(?:\w+\s+){0,2}$",
    re.I,
)


def flag_contradicted_claims(
    text: str, indicators: TechnicalIndicators
) -> list[str]:
    """Flag prose that contradicts a relation computed from the indicators.

    Deliberately narrower than the numbers guard, and a different kind of
    check: it does not ask whether a number is real, it asks whether a claim
    is true. That only works where the ground truth is unambiguous, so it
    covers one family of statement — price versus a moving average — which is
    where the observed error occurred and where "above" and "below" have no
    room for interpretation.

    Flags rather than blocks, like the numbers guard: a false positive should
    cost a reviewer a glance, not a run.
    """
    close = indicators.last_close
    flags: list[str] = []

    for match in _PRICE_VS_MA.finditer(text):
        direction, period = match.group(1).lower(), match.group(2)
        value = getattr(indicators, f"sma_{period}", None)
        if value is None:
            continue  # no such indicator (or not computed) — nothing to check
        if _MA_IS_SUBJECT.search(text[: match.start()]):
            continue  # comparing two averages, not price against one

        actually_above = close > value
        if actually_above == (direction == "above"):
            continue

        flags.append(
            f"claims price is {direction} the {period}-day average, but last "
            f"close {close:.2f} is {'above' if actually_above else 'below'} "
            f"sma_{period} {value:.2f}"
        )

    return flags


# A number with an optional sign, where '-' is read as a sign only if the
# preceding character can't make it a separator instead: a digit or '.' means
# a numeric range ("318.73-352.11"), a '%' means a percentage range
# ("88%-89%"). Both are ordinary phrasings, and reading their hyphen as a
# minus turns the second endpoint into a negative number that matches no
# indicator value.
_SIGNED_NUMBER = r"(?<![\d.%])-?\d+\.?\d*"

# Window labels, stripped before the value-check because they name a period
# rather than report a measurement. Two spellings, both seen live:
#
#   "the 200-day average"            -> the plain compound
#   "the 50- and 200-day averages"   -> a suspended hyphen, where the noun is
#                                       carried by the second term only
#
# The second cost a false positive: "200-day" was stripped, the dangling "50-"
# was not, and the orphaned 50 matched no indicator value. The lookahead is
# restricted to a following conjunction so this only ever fires on a genuine
# suspended compound — matching any digit-hyphen-space would eat the first
# endpoint of a spaced range like "318.73 - 352.11".
_PERIOD_LABEL = re.compile(r"\b\d+-day\b|\b\d+-(?=\s+(?:and|or|to)\s)")


def _flag_unmatched_numbers_against(text: str, known_values: list[float]) -> list[str]:
    """Cheap guard, not a full verifier: extract numbers mentioned in the interpretation
    and check each is within rounding tolerance of some value actually in `indicators`.
    Flags (doesn't block) anything that doesn't match — surfaced in TechnicalReport for
    human review, same spirit as the 'Unverified Figures' section in memo_verifier.

    Still a review signal rather than an auto-reject: it can produce false positives on
    narrative numbers that reference thresholds rather than indicator values themselves.
    The RSI band edges were the recurring instance of that and are now exempted near
    RSI context (see _is_rsi_band_reference) — they were unmatchable by construction,
    and the derived-relations block made the model restate them on every run, so the
    guard was reporting the same two numbers forever. Other threshold vocabulary
    (Bollinger deviations, MACD zero-line talk) has not shown up in practice and is
    left alone rather than pre-emptively exempted.

    Three known transformations are normalized before flagging, each patched from a
    real false positive rather than designed upfront — coverage is only as good as
    the phrasing actually tested, not something derivable from first principles:

    1. Period-descriptor phrases ("50-day", "200-day") are stripped before scanning,
       since those are label numbers (the SMA/RSI window length), not data values. This
       narrows the false-positive surface but opens a corresponding gap: a fabricated
       period ("the 55-day average") would slip through unflagged, since it's stripped
       before the value-check ever sees it. See
       test_flag_unmatched_numbers_does_not_catch_fabricated_period_label for that
       documented boundary.
    2. "N%" mentions are checked against known_values/100 as well as known_values
       directly — confirmed necessary when volume_vs_20d_avg=0.529 was faithfully
       reported as "53%" and would otherwise have been flagged as fabricated.
    3. "N% above/below" mentions are checked against (known_value - 1) * 100 —
       a distinct transform from #2: "22% above the 20-day average" describes a
       *delta* from a ratio-type value (volume_vs_20d_avg=1.2153 -> (1.2153-1)*100
       = 21.5% =~ "around 22"), not the raw ratio-as-percentage. Matched (and
       consumed) before the general percent pattern so the two don't collide.
       Both endpoints of a range ("21%-22% above the 20-day average") are
       captured by the one match, because only the endpoint touching the
       keyword carries the "above/below" context — matching it alone leaves
       the other orphaned for the general percent rule, which then tests a
       delta as though it were a ratio and flags faithful text.

    A leading '-' counts as a sign only where it can't be a range separator
    (_SIGNED_NUMBER). Negative indicator values are ordinary — a bearish
    macd_histogram of -1.2158 gets reported as "around -1.22" — so the sign
    has to parse, but reading every hyphen as one turned a faithful
    "318.73-352.11" band into a fabricated "-352.11" and flagged a real
    bb_upper value. This is a parsing fix, not a tolerance one: unlike the
    threshold false positives above ("RSI above 70"), the number was a
    genuine indicator value that the scanner mangled before comparing it.
    """
    # Normalize U+2212 MINUS SIGN to ASCII before anything reads a sign. The
    # model writes typographic minus roughly one run in four — "negative
    # histogram of −3.23" — and _SIGNED_NUMBER only knows the ASCII hyphen, so
    # a faithful -3.2298 parsed as positive 3.23 and matched nothing. Python
    # agrees on the narrower alphabet: float("−3.23") raises.
    #
    # Only U+2212, not the dashes. En dash is the conventional range separator
    # ("318.73–352.11"), and the whole reason _SIGNED_NUMBER carries a
    # lookbehind is that reading a separator as a sign turned a real Bollinger
    # band into a fabricated negative.
    text = text.replace("−", "-")
    text_no_periods = _PERIOD_LABEL.sub("", text)

    flagged: list[str] = []

    above_below_pattern = re.compile(
        rf"({_SIGNED_NUMBER})%(?:\s*-\s*({_SIGNED_NUMBER})%)?\s*(?:above|below)"
    )
    for endpoints in above_below_pattern.findall(text_no_periods):
        for m in (e for e in endpoints if e):
            delta_pct = float(m)
            if not any(abs(delta_pct - (kv - 1) * 100) <= max(1.0, abs(kv) * 2) for kv in known_values):
                flagged.append(f"{m}% above/below")
    text_no_above_below = above_below_pattern.sub("", text_no_periods)

    percent_pattern = re.compile(rf"({_SIGNED_NUMBER})%")
    for m in percent_pattern.findall(text_no_above_below):
        ratio = float(m) / 100
        if not any(abs(ratio - kv) <= max(0.01, abs(kv) * 0.02) for kv in known_values):
            flagged.append(f"{m}%")
    text_no_percents = percent_pattern.sub("", text_no_above_below)

    for match in re.finditer(_SIGNED_NUMBER, text_no_percents):
        m = match.group()
        val = float(m)
        if any(abs(val - kv) <= max(0.5, abs(kv) * 0.02) for kv in known_values):
            continue
        if _is_rsi_band_reference(val, text_no_percents, match.start(), match.end()):
            continue
        flagged.append(m)

    return flagged



def _flag_unmatched_numbers(text: str, indicators: TechnicalIndicators) -> list[str]:
    """Unchanged public signature — the Phase 3 tests keep passing untouched.

    The list-taking form above was extracted so Phase 5's debate guard can
    reuse the percent transforms over numbers scraped out of the analyst
    reports. The alternative — handing this function a fake object with a
    `model_dump()` — would make the debate depend on a duck-typed shim that
    no test covers.
    """
    return _flag_unmatched_numbers_against(
        text,
        [v for v in indicators.model_dump().values() if isinstance(v, (int, float))],
    )


# The conventional RSI band edges. These are constants of the indicator, not
# values read off it, so they can never appear in `indicators` and were
# guaranteed to flag — "RSI above 70" was documented as a known false positive
# from the start, and adding a relations block that says "NEITHER overbought
# nor oversold (between 30 and 70)" made the model echo them every run.
#
# Only 30 and 70. The 20/80 variant exists, but every value added here is a
# value the guard can no longer catch anywhere it appears, and 20 and 80 are
# far likelier to collide with a genuine price or indicator reading.
_RSI_BAND_VALUES = {30.0, 70.0}

# Required nearby for the exemption to apply, so 30 and 70 stay checkable
# everywhere else — a fabricated "P/E ratio of 30" is still flagged.
_RSI_CONTEXT = re.compile(r"\brsi\b|overbought|oversold", re.I)

# Asymmetric because the giveaway usually precedes the number ("RSI ...
# between 30 and 70") and only sometimes follows it ("70, neither overbought
# nor oversold"). Wide enough to reach back past "sits comfortably in neutral
# territory between 30 and", narrow enough not to borrow an RSI mention from
# a neighbouring sentence.
_RSI_LOOKBEHIND = 90
_RSI_LOOKAHEAD = 60


def _is_rsi_band_reference(value: float, text: str, start: int, end: int) -> bool:
    """True when a number is one of the RSI band edges, used as a threshold.

    Scoped by proximity rather than exempted outright: this narrows the
    guard's blind spot to "30 or 70 written within a sentence's reach of the
    word RSI", instead of blinding it to those two values everywhere.
    """
    if value not in _RSI_BAND_VALUES:
        return False
    window = text[max(0, start - _RSI_LOOKBEHIND) : end + _RSI_LOOKAHEAD]
    return _RSI_CONTEXT.search(window) is not None


def _format_technical_markdown(report: TechnicalReport) -> str:
    ind = report.indicators
    lines = [
        f"# {report.ticker} — Technical Analysis",
        f"**Date:** {report.as_of_date}",
        f"**Source:** {report.data_source} ({report.bars_used} daily bars)",
        "",
        "## Indicators",
        "",
        "| Indicator | Value |",
        "|---|---|",
        f"| SMA(50) | {ind.sma_50} |",
        f"| SMA(200) | {ind.sma_200} |",
        f"| RSI(14) | {ind.rsi_14} |",
        f"| MACD | {ind.macd} |",
        f"| MACD Signal | {ind.macd_signal} |",
        f"| MACD Histogram | {ind.macd_histogram} |",
        f"| Bollinger Upper | {ind.bb_upper} |",
        f"| Bollinger Mid | {ind.bb_mid} |",
        f"| Bollinger Lower | {ind.bb_lower} |",
        f"| Last Close | {ind.last_close} |",
        f"| Volume vs 20d Avg | {ind.volume_vs_20d_avg} |",
        "",
        "## Interpretation",
        "",
        report.interpretation,
    ]
    if report.interpretation_flagged_claims:
        lines += [
            "",
            "## Contradicted Claims",
            "",
            "Statements above that contradict a relation computed from the "
            "indicators. Unlike a flagged number, these use real values to say "
            "something false — treat the interpretation as unreliable here.",
            "",
        ] + [f"- {c}" for c in report.interpretation_flagged_claims]
    if report.interpretation_flagged_numbers:
        lines += [
            "",
            "## Flagged Numbers",
            "",
            "Numbers in the interpretation that could not be matched back to a "
            "retrieved indicator value. Review before relying on them.",
            "",
            ", ".join(report.interpretation_flagged_numbers),
        ]
    return "\n".join(lines)


def save_technical_report(report: TechnicalReport, cost_usd: float | None = None) -> Path:
    content = _format_technical_markdown(report)
    return _save_output(content, report.ticker.upper(), "technical", cost_usd=cost_usd)
