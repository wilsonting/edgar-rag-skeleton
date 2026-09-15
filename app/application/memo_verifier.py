"""Verify the final memo against everything the tools returned this run."""

from app.application.citation_verifier import verify_answer
from app.application.verdict_consistency import verdict_warning


def verify_memo(memo: str, provenance_corpus: str,
                computed_values: list[float],
                rejected_calcs: list[dict] | None = None) -> str:
    # Read from the memo alone, before the provenance checks and outside
    # their early returns: a memo can be perfectly sourced, or have no
    # provenance at all, and still contradict itself about what it concludes.
    verdict_lines = verdict_warning(memo)

    def _appended(*sections: list[str]) -> str:
        body = [x for s in sections for x in s]
        return memo + "\n".join(["", "---"] + body) if body else memo

    if not provenance_corpus.strip():
        return _appended(verdict_lines)

    report = verify_answer(memo, {0: provenance_corpus},
                           computed_values=computed_values,
                           rejected_calcs=rejected_calcs)
    if report.ok and not report.flagged and not report.underived:
        return _appended(verdict_lines)

    lines = ["", "---", ""] + verdict_lines

    if report.unverified:
        lines += [
            "## Unverified Figures and Quotations", "",
            "The following appear in this memo but were not returned by any "
            "tool during this run. Verify against the filing before relying "
            "on them.", "",
        ]
        for f in report.unverified:
            label = "quote" if f.kind == "quote" else "figure"
            lines.append(f"- **{label}:** `{f.value}`")
            lines.append(f"  - context: …{f.context.strip()}…")

    if report.flagged:
        lines += [
            "", "## Unbacked Derivations", "",
            "The following figures match a `calculate()` call that was "
            "rejected during this run and never successfully retried. The "
            "number itself may be correct, but no passing tool call in "
            "this run's trace validates how it was derived — re-derive it "
            "with a passing calculate call before relying on it.", "",
        ]
        for f in report.flagged:
            lines.append(f"- **figure:** `{f.value}`")
            lines.append(f"  - context: …{f.context.strip()}…")

    if report.underived:
        lines += [
            "", "## Underived Arithmetic", "",
            "The following figures were not returned by any tool, but each "
            "equals a sum or difference of other figures in this memo that "
            "were independently verified — a total stated in prose instead "
            "of run through calculate(). The inputs are real; only this "
            "arithmetic step is unvalidated. Lower risk than the figures "
            "above — re-derive with calculate() to confirm.", "",
        ]
        for f in report.underived:
            lines.append(f"- **figure:** `{f.value}`")
            lines.append(f"  - context: …{f.context.strip()}…")

    return memo + "\n".join(lines)