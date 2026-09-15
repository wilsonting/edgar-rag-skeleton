"""Deciding whether a number appears in text — token-anchored, one implementation.

Moved out of citation_verifier.py so the `calculate` provenance checks in
app/agent/tools.py use the same rule. They had their own substring version
(`any(v in corpus for v in variants)`), which is exactly the weakness the
verifier's docstring records fixing: a figure "verified" by occurring inside
an unrelated larger number. Any two-digit integer was "retrieved" because it
sits inside a year ("12" in "2012"), and "7.4" because of "17.45".

The rule: a number matches a standalone numeric token in the text — exactly,
as an integer truncation of a more precise figure (text "1,364.1", number
1364), or as a rounding of it within half a step of the number's own
displayed precision (text "10874.36", number "10,874.4").

The debate and technical guards (debate_port._flag_debate_numbers,
technical_interpreter_port._flag_unmatched_numbers_against) still carry their
own matchers, with deliberately different tolerances; folding them in is a
separate change (docs/code_review.md, Medium #6).
"""

from __future__ import annotations

import re

# Numbers with optional thousands separators and decimals: 27,558.5  1364.1  118
NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


def number_variants(raw: str) -> set[str]:
    """
    Every string form a filing might use for the number an answer writes
    as `raw`. Filings write 27,558.5; answers write 27558.5 or 27,558.5.
    Filings also write 1,364.1 where an answer may write 1364.
    """
    bare = raw.replace(",", "")
    variants = {raw, bare}

    try:
        val = float(bare)
    except ValueError:
        return variants

    # comma-grouped form
    if val == int(val):
        variants.add(f"{int(val):,}")
        variants.add(str(int(val)))
    else:
        variants.add(f"{val:,}")
        # trailing-zero and one-decimal forms: 1364.10 -> 1364.1
        variants.add(f"{val:,.1f}")
        variants.add(f"{val:.1f}")
        variants.add(f"{val:,.2f}")

    return {v for v in variants if v}


def number_tokens(text: str) -> list[str]:
    """Every standalone numeric token in `text`, as `NUMBER_RE` finds it.

    finditer already tokenizes atomically — on "$3,420.5 thousand" it
    yields the single token "3,420.5", never a spurious "420.5" — so this
    exists to make that tokenization explicit and reusable, not to add new
    parsing behavior."""
    return [m.group().rstrip(".") for m in NUMBER_RE.finditer(text) if m.group().rstrip(".")]


def half_step(raw: str) -> float:
    """Half of one step in `raw`'s last displayed decimal place — the
    largest distance a source figure can sit from `raw` while still
    legitimately rounding to it. "877.4" -> 0.05; "615" -> 0.5."""
    bare = raw.replace(",", "")
    decimals = len(bare.split(".")[1]) if "." in bare else 0
    return 0.5 * 10 ** -decimals


def matches_token(raw: str, variants: set[str], token: str) -> bool:
    """True if the number `raw` matches this single corpus token —
    exactly, as an integer truncation of a more precise filing figure
    (filing writes "1,364.1", memo writes "1364"), or as a rounding of it
    at the memo's own displayed precision (extract_metrics returns
    10874.36, memo writes "10,874.4" — within half a step of the memo's
    last decimal, so a legitimate restatement, not an invention).

    Deliberately NOT a substring check (`variant in token` or `variant in
    text`): that allowed a memo figure to "verify" merely by occurring
    inside an unrelated, larger corpus number that happens to contain the
    same digits — e.g. a fabricated "420.5" matched because "3,420.5"
    (a different figure entirely) appeared somewhere in the corpus.
    Matching against whole tokens, with truncation and rounding as the
    only numeric slop — both bounded by the memo's displayed precision —
    closes that gap while keeping the restatement cases the substring
    check was originally added for.
    """
    bare_token = token.replace(",", "")
    for v in variants:
        bare_v = v.replace(",", "")
        if v == token or bare_v == bare_token:
            return True
        if bare_token.startswith(bare_v + "."):
            return True
    try:
        return abs(float(bare_token) - float(raw.replace(",", ""))) <= half_step(raw)
    except ValueError:
        return False


def raw_form(value: float) -> str:
    """The shortest faithful string for a float, as a model would have
    written it: 11474.0 -> "11474", 64896.464 -> "64896.464". Its displayed
    precision is what bounds the rounding slop in `matches_token`."""
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def value_spans(value: float, text: str) -> list[tuple[int, int]]:
    """(start, end) of every numeric token in `text` that `value` matches."""
    raw = raw_form(value)
    variants = number_variants(raw)
    spans = []
    for m in NUMBER_RE.finditer(text):
        token = m.group().rstrip(".")
        if token and matches_token(raw, variants, token):
            spans.append((m.start(), m.start() + len(token)))
    return spans


def value_in_text(value: float, text: str) -> bool:
    return bool(value_spans(value, text))
