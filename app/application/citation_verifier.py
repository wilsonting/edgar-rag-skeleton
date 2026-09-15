"""
Citation verification.

Checks that every numeric literal and quoted string in a generated answer
actually appears in at least one of the chunks that were retrieved to
produce it.

SCOPE — read this before trusting it:
  Catches   : fabricated figures, fabricated quotations.
  Misses    : fabricated causation ("new Eurobond issuance drove the
              increase"), and any claim with no literal to check.

The verifier answers "does this number exist in the source material", not
"is this claim true". Those are different questions and only the first is
mechanically decidable.

Wrong fiscal-year attribution of a real figure used to be an unmitigated
miss here too — a value from one period, mislabeled and used as another
period's input, is individually real and passes this check every time.
That specific case is now caught one layer upstream, in
validate_calculate_inputs (app/agent/tools.py, check 4): every calculate()
input declares a fiscal_period, and that check confirms the declared
period's year actually appears near where the value was retrieved. This
verifier still can't catch a mislabeled figure that never went through
calculate() — e.g. a raw retrieved number stated directly in prose without
a computation. That gap remains open.

A prior version matched corpus figures by raw substring search
(`variant in text`), which let a fabricated number "verify" merely by
occurring inside an unrelated, larger corpus number — a made-up "$420.5M"
passed because "$3,420.5" (a different figure) appeared somewhere in the
retrieved text. Matching is now token-anchored (`_matches_corpus_token`):
a memo number must equal a whole corpus number token, not just be found
somewhere inside one. The one legitimate case the substring check
existed for — a memo truncating a filing's more precise decimal, e.g.
memo "1364" for filing "1,364.1" — is still handled explicitly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations, product

from app.application.number_matching import (
    NUMBER_RE as _NUMBER_RE,
    half_step as _half_step,
    matches_token as _matches_corpus_token,
    number_tokens as _corpus_number_tokens,
    number_variants as _number_variants,
)


# ---------------------------------------------------------------------------
# What we pull out of an answer
# ---------------------------------------------------------------------------


def _extract_quotes(text: str) -> list[tuple[str, int]]:
    """
    Pair quote characters sequentially (1st-2nd, 3rd-4th, ...) and return
    spans of >= 4 words with their start offsets.

    A regex like ["“]([^"”]{15,}?)["”] pairs the CLOSING quote of a short
    quotation with the OPENING quote of the next one, capturing the prose
    between them. Observed: 'measures being "put in place" (FY2025 10-K) to
    being "completed"' yielded ' (FY2025 10-K) to being ' as a quotation.
    """
    positions = [m.start() for m in re.finditer(r"[\"\u201c\u201d]", text)]
    out: list[tuple[str, int]] = []
    for i in range(0, len(positions) - 1, 2):
        start, end = positions[i], positions[i + 1]
        span = text[start + 1 : end]
        if len(span.split()) >= 4 and len(span) >= 15:
            out.append((span, start))
    return out

# Figures that are almost always structural rather than sourced: years,
# percentages of the model's own construction, list numbering, small counts.
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")


@dataclass
class Finding:
    kind: str                  # "number" | "quote"
    value: str                 # as it appeared in the answer
    context: str               # surrounding text, for the report
    matched_chunk_id: int | None = None


@dataclass
class VerificationReport:
    verified: list[Finding] = field(default_factory=list)
    unverified: list[Finding] = field(default_factory=list)
    skipped: list[Finding] = field(default_factory=list)
    # Numbers that exist in the source material (so they're NOT "unverified")
    # but that also match a calculate() call this run rejected and the model
    # never successfully retried — the number may well be correct, but no
    # passing tool call in this run's trace backs how it was derived.
    flagged: list[Finding] = field(default_factory=list)
    # Numbers that never matched the corpus or a calculate() result directly,
    # but that equal a sum/difference of OTHER numbers in the memo that did
    # verify — a total the model wrote in prose (e.g. "$67,566 (1,271 +
    # 66,295)") instead of running through calculate(). Every operand is
    # real; only the arithmetic step is unvalidated. Kept out of `unverified`
    # so a reader isn't trained to skim past genuine fabrications sitting in
    # the same list as trivial, correctly-derived sums.
    underived: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unverified

    def summary(self) -> str:
        lines = [
            f"Citation check: {len(self.verified)} verified, "
            f"{len(self.unverified)} UNVERIFIED, {len(self.skipped)} skipped, "
            f"{len(self.flagged)} flagged (unbacked derivation), "
            f"{len(self.underived)} underived (arithmetic on verified inputs)"
        ]
        for f in self.unverified:
            lines.append(f"  UNVERIFIED {f.kind}: {f.value}")
            lines.append(f"      context: …{f.context}…")
        for f in self.flagged:
            lines.append(f"  UNBACKED {f.kind}: {f.value}")
            lines.append(f"      context: …{f.context}…")
        for f in self.underived:
            lines.append(f"  UNDERIVED {f.kind}: {f.value}")
            lines.append(f"      context: …{f.context}…")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Normalization — the part that decides whether this is useful or noisy
# ---------------------------------------------------------------------------

def _normalize_text(s: str) -> str:
    """Collapse whitespace and smart quotes so quote matching survives
    the line breaks and typography of parsed filing text."""
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u2014", "-").replace("\u2013", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


def _should_skip_number(raw: str, context: str) -> bool:
    bare = raw.replace(",", "")

    # Years are structural, and appear in citation tags the model constructs.
    if _YEAR_RE.match(bare):
        return True

    # Section references: "Item 1A", "Item 7". Only the section number
    # itself is skipped — not every number that happens to sit near one.
    if re.search(rf"Item\s+{re.escape(raw)}\b", context):
        return True

    try:
        val = float(bare)
    except ValueError:
        return True

    # Small integers are almost always enumeration, section numbers, or
    # counts the model constructed ("three risk factors", "1.", "top 5").
    if val == int(val) and val < 100:
        return True

    return False


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _computed_forms(values) -> set[str]:
    """String forms a calculate() result may take in the answer text."""
    out: set[str] = set()
    for v in values or []:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        out.update({f"{f:.1f}", f"{f:.2f}", str(round(f, 1)), str(round(f, 2))})
    return out


def _rejected_match_tolerance(value: float) -> float:
    """How far a memo figure may sit from a rejected calculate() result and
    still count as the same derivation.

    Deliberately numeric, not a set of pre-rounded strings: an earlier
    version matched by intersecting rounded-string forms of both sides, which
    missed a real case — a rejected expression whose true result was
    18.1503 (rounds to 18.2) was stated in the memo as 18.1, its own
    rounding error, off the true value by 0.05. No string form of 18.1503
    is "18.1", so an exact-string match can never catch a value the model
    itself rounded wrong — only a numeric distance check can. Ratios and
    percentages are reported to 1 decimal, so a flat floor of 0.1 (a full
    step in the last displayed digit) plus a small relative term covers
    that rounding slop without being so loose it matches unrelated figures.
    """
    return max(0.1, abs(value) * 0.01)


def _split_retried_rejected(
    rejected: list[dict] | None, computed_values: list[float] | None
) -> list[dict]:
    """Rejected attempts whose result was never matched by a later passing
    calculate() call — i.e. still lacking a validated derivation."""
    computed = list(computed_values or [])
    out = []
    for r in rejected or []:
        try:
            rv = float(r.get("value"))
        except (TypeError, ValueError):
            continue
        tol = _rejected_match_tolerance(rv)
        if any(abs(rv - cv) <= tol for cv in computed):
            continue
        out.append(r)
    return out


def _find_rejected_match(value: float, rejected: list[dict]) -> dict | None:
    """The rejected-and-unretried record whose true result `value` matches,
    within `_rejected_match_tolerance` — or None."""
    for r in rejected:
        try:
            rv = float(r["value"])
        except (TypeError, ValueError):
            continue
        if abs(rv - value) <= _rejected_match_tolerance(rv):
            return r
    return None



_CORPUS_NUM_RE = re.compile(r"-?\(?[\d,]+\.?\d*\)?")


def _corpus_values(corpus: str) -> list[float]:
    """Every number in the corpus, as floats. Parenthesised = negative."""
    out: list[float] = []
    for m in _CORPUS_NUM_RE.finditer(corpus):
        s = m.group().strip()
        neg = s.startswith("(") and s.endswith(")")
        s = s.strip("()").replace(",", "")
        if not s or s in {".", "-"}:
            continue
        try:
            v = float(s)
        except ValueError:
            continue
        out.append(-v if neg else v)
    return out


def _matches_with_scale(value: float, raw: str, corpus_values: list[float]) -> bool:
    """
    True if `value` matches a corpus figure directly or after a unit change.

    Filings that report in thousands are routinely restated in millions in a
    memo: the corpus holds 877,433 and the memo says 877.4. Comparison is
    numeric, and the tolerance is direction-aware:

    - corpus figure scaled DOWN to the memo's units: the memo is a rounded
      restatement of a more precise source figure, so the tolerance is the
      memo's own rounding slop — half a step in its last displayed decimal
      (`_half_step`). 877,433/1000 = 877.433 rounds to the memo's 877.4.
    - memo figure scaled DOWN to a corpus figure: the memo is claiming MORE
      precision than the corpus figure carries, which a rounded restatement
      can never legitimately do — so only an essentially exact match (tight
      relative tolerance, no absolute floor) counts.

    An earlier version used max(0.05, x*0.0005) symmetrically in both
    directions. After dividing by 1000, that 0.05 absolute floor is a ±50
    window on the original number — wide enough that six fabricated memo
    figures ($11,474.4M operating cash flow, $600.0M capex, ...) all
    "verified" against nothing but citation similarity scores (516.5/1000 ≈
    sim=0.516) and unrelated one-decimal percentages in prose (11,474.4/1000
    ≈ "11.5%"). In a dense corpus nearly any 3-5 digit number collided with
    something.
    """
    av = abs(value)
    half = _half_step(raw)
    for c in corpus_values:
        ac = abs(c)
        if ac == av:
            return True
        for scale in (1000.0, 1_000_000.0):
            # corpus in thousands, memo in millions: memo may be rounded
            if ac and abs(ac / scale - av) <= half:
                return True
            # memo in thousands, corpus in millions: memo may not invent
            # precision beyond the corpus figure — near-exact only
            if av and ac and abs(av / scale - ac) <= ac * 0.0005:
                return True
    return False


def _derivation_tolerance(target: float) -> float:
    """How far a candidate sum/difference may sit from `target` and still
    count as the same arithmetic — mirrors `_rejected_match_tolerance`: a
    full step in the last displayed digit, plus a small relative term."""
    return max(0.1, abs(target) * 0.001)


def _try_derive(target: float, candidates: list[float], tol: float) -> bool:
    """True if some subset of 2-4 `candidates`, each independently added or
    subtracted, sums to `target` within `tol`. Bounded to small subsets —
    memo totals are sums of a handful of stated components, not arbitrary
    combinations, and this keeps the search cheap."""
    n = len(candidates)
    if n < 2:
        return False
    for size in range(2, min(n, 4) + 1):
        for combo in combinations(candidates, size):
            for signs in product((1, -1), repeat=size):
                total = sum(s * c for s, c in zip(signs, combo))
                if abs(total - target) <= tol:
                    return True
    return False


def verify_answer(
    answer: str,
    chunk_texts: dict[int, str],
    computed_values: list[float] | None = None,
    rejected_calcs: list[dict] | None = None,
) -> VerificationReport:
    """
    answer          : the generated answer text
    chunk_texts     : {chunk_id: full chunk content} for every chunk
                      retrieved to produce this answer
    computed_values : results returned by calculate() during this run.
                      Without these, every legitimately computed ratio is
                      reported unverified and the report becomes noise.
    rejected_calcs  : calculate() calls rejected this run and never
                      successfully retried — each a dict with `value`
                      (what the expression would have evaluated to),
                      `reason`, and `expression`. A number matching one of
                      these is real (it'll usually also verify against the
                      corpus) but its derivation was never validated by a
                      passing tool call; recorded in `report.flagged`
                      rather than `report.unverified`.

    Returns a VerificationReport. `unverified` is what needs a human look.
    """
    report = VerificationReport()
    computed = _computed_forms(computed_values)
    # A rejected attempt later backed by a passing calculate() call (same
    # numeric result) is validated, regardless of whether the caller already
    # filtered it out — don't rely solely on the caller's bookkeeping.
    unretried_rejected = _split_retried_rejected(rejected_calcs, computed_values)

    normalized_chunks = {cid: _normalize_text(t) for cid, t in chunk_texts.items()}
    chunk_tokens = {cid: _corpus_number_tokens(t) for cid, t in chunk_texts.items()}
    _corpus_nums = _corpus_values("\n".join(chunk_texts.values()))

    # --- numbers -----------------------------------------------------------
    # Provisionally-unverified numbers, held back from report.unverified with
    # their position so a second pass (below) can check whether each is a
    # sum/difference of other numbers the memo states nearby that DID verify
    # — a total written in prose instead of run through calculate().
    provisional_unverified: list[tuple[Finding, int]] = []
    seen_numbers: set[str] = set()
    for m in _NUMBER_RE.finditer(answer):
        raw = m.group().rstrip(".")
        if not raw or raw in seen_numbers:
            continue
        seen_numbers.add(raw)

        start = max(0, m.start() - 45)
        context = answer[start : m.end() + 45].replace("\n", " ")

        if _should_skip_number(raw, context):
            report.skipped.append(Finding("number", raw, context))
            continue

        bare = raw.replace(",", "")

        # Matches a calculate() call that was rejected and never
        # successfully retried: real (or near-real) number, unvalidated
        # derivation. This needs its own channel rather than reusing
        # `unverified` — the number will usually verify against the corpus
        # (that's exactly how this slips through unnoticed otherwise). It
        # also supersedes the verified/unverified classification entirely:
        # the flagged entry carries the rejection reason, and listing the
        # same figure in both report sections reads as two separate
        # problems when it's one.
        try:
            rejected_hit = _find_rejected_match(float(bare), unretried_rejected)
        except ValueError:
            rejected_hit = None
        if rejected_hit is not None:
            report.flagged.append(Finding(
                "derivation", raw,
                f"{context} [matches a calculate() call rejected for: "
                f"{rejected_hit['reason'][:120]} — never successfully "
                f"retried]",
            ))
            continue

        # A figure matching a calculate() result is verified as computed,
        # not fabricated. chunk id -1 marks "produced by the calculator".
        if bare in computed or _computed_forms([bare]) & computed:
            report.verified.append(Finding("computed", raw, context, -1))
            continue

        variants = _number_variants(raw)
        hit = None
        for cid, tokens in chunk_tokens.items():
            if any(_matches_corpus_token(raw, variants, tok) for tok in tokens):
                hit = cid
                break
        if hit is None:
            # Fall back to scale-aware numeric comparison (thousands vs millions)
            try:
                if _matches_with_scale(float(bare), raw, _corpus_nums):
                    hit = -2
            except ValueError:
                pass

        f = Finding("number", raw, context, hit)
        if hit is not None:
            report.verified.append(f)
        else:
            provisional_unverified.append((f, m.start()))

    # Second pass: a provisionally-unverified number may be a sum/difference
    # of other numbers the memo states nearby that already verified — e.g.
    # "$67,566 (comprised of $1,271 and $66,295)" where the two components
    # are real retrieved figures but the total itself was never retrieved or
    # run through calculate(). Only numbers that (a) sit in a wide window
    # around the target AND (b) are themselves independently verified count
    # as candidates — a number merely appearing nearby proves nothing on its
    # own, which is what keeps this from rescuing genuine fabrications.
    verified_bare = {
        f.value.replace(",", "") for f in report.verified if f.kind in ("number", "computed")
    }
    for f, pos in provisional_unverified:
        window = answer[max(0, pos - 200) : pos + 200]
        candidates = []
        for cm in _NUMBER_RE.finditer(window):
            cand_raw = cm.group().rstrip(".")
            cand_bare = cand_raw.replace(",", "")
            if cand_bare == f.value.replace(",", ""):
                continue
            if cand_bare in verified_bare:
                try:
                    candidates.append(float(cand_bare))
                except ValueError:
                    continue
        try:
            target = float(f.value.replace(",", ""))
        except ValueError:
            report.unverified.append(f)
            continue
        if _try_derive(target, candidates, _derivation_tolerance(target)):
            report.underived.append(f)
        else:
            report.unverified.append(f)

    # --- quotes ------------------------------------------------------------
    for quoted, qstart in _extract_quotes(answer):
        needle = _normalize_text(quoted)
        start = max(0, qstart - 30)
        context = answer[start : qstart + len(quoted) + 32].replace("\n", " ")

        hit = None
        for cid, text in normalized_chunks.items():
            if needle in text:
                hit = cid
                break

        f = Finding("quote", quoted[:70], context, hit)
        (report.verified if hit is not None else report.unverified).append(f)

    return report