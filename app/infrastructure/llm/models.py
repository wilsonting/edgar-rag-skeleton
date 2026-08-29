"""Which model serves which role — the whole answer, in one place.

Every LLM role in the pipeline reads its model from `.env`. Roles were
previously scattered: five read a dedicated variable, five more were pinned
to whatever `LLM_CLAUDE_MODEL` said with no way to move them independently,
and `.env` carried two variables (`RISK_MODEL`, `RISK_JUDGE_MODEL`) that no
code read at all — so changing them looked like it worked and did nothing.

`LLM_CLAUDE_MODEL` stays the project-wide default and is the only required
variable: every other role falls back to it when its own variable is unset.
One knob still configures the whole pipeline; ten knobs are there when a
role needs to differ.

    uv run python -m app.infrastructure.llm.models

prints the resolved table — role, variable, model, provider, priced — which
is the fastest way to answer "what is this run actually going to use".
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from app.infrastructure.llm.pricing import MODEL_PRICING

# The project-wide default. Required: there is no sensible built-in model id
# to fall back to, and guessing one would mean spending money on a model
# nobody chose.
DEFAULT_MODEL_ENV = "LLM_CLAUDE_MODEL"


@dataclass(frozen=True)
class ModelRole:
    key: str
    env: str
    what: str


# Order is the order a run uses them, which is also the order the table
# prints in.
ROLES: tuple[ModelRole, ...] = (
    ModelRole("agent", DEFAULT_MODEL_ENV,
              "research agent loop, and the fundamentals node that runs it"),
    ModelRole("answer", "LLM_ANSWER_MODEL",
              "answer generation behind POST /ask"),
    ModelRole("decomposer", "LLM_DECOMPOSER_MODEL",
              "query decomposition"),
    ModelRole("extraction", "LLM_EXTRACTION_MODEL",
              "financial-metric extraction"),
    ModelRole("news_digest", "TRADING_NEWS_DIGEST_MODEL",
              "news digest"),
    ModelRole("technical", "TRADING_TECHNICAL_MODEL",
              "technical-indicator interpretation"),
    ModelRole("debate", "TRADING_DEBATE_MODEL",
              "bull/bear debate turns"),
    ModelRole("risk_panel", "TRADING_RISK_MODEL",
              "risk panel personas"),
    ModelRole("research_manager", "TRADING_RESEARCH_MANAGER_MODEL",
              "research manager synthesis"),
    ModelRole("risk_judge", "TRADING_RISK_JUDGE_MODEL",
              "risk judge, final verdict"),
)

_BY_KEY = {role.key: role for role in ROLES}


def model_for(key: str) -> str:
    """The model id for a role, read live from the environment.

    Read live rather than snapshotted at import so a test or a script can
    move one role with `monkeypatch.setenv` without reloading modules. The
    ports still bind a module-level constant from this at import, which is
    what every log line and cost event reports.
    """
    role = _BY_KEY.get(key)
    if role is None:
        raise KeyError(f"unknown model role {key!r} — known roles: {sorted(_BY_KEY)}")

    explicit = os.getenv(role.env)
    if explicit:
        return explicit
    if role.env == DEFAULT_MODEL_ENV:
        # The one role with nothing to fall back to.
        return os.environ[DEFAULT_MODEL_ENV]
    return os.environ[DEFAULT_MODEL_ENV]


def model_env_vars() -> list[str]:
    """Every variable that actually selects a model.

    `scripts/run_p9_battery.py` records these with each run so a rerun can
    be checked against the configuration that produced the original — which
    only works if the list is the code's, not a hand-maintained copy that
    drifts.
    """
    return [role.env for role in ROLES]


def configured_models() -> dict[str, str]:
    return {role.key: model_for(role.key) for role in ROLES}


def warn_if_unpriced(model: str, label: str, budget_usd: float | None = None) -> None:
    """Say once, at import, that a role's model has no pricing.

    Not fatal, but worth shouting about: an unpriced model makes every call
    log a cost of `None`, `None` sums to 0.00, and a budget assertion
    comparing 0.00 against a ceiling can never fire. The guard would be
    silently absent rather than merely loose — which is indistinguishable
    from a run that came in under budget.
    """
    if model in MODEL_PRICING:
        return
    budget = (
        f"the ${budget_usd:.2f} budget assertion cannot fire"
        if budget_usd is not None
        else "nothing will constrain what this role spends"
    )
    print(
        f"[{label}] WARNING: no pricing configured for {model} — its calls "
        f"will log cost as null and {budget}. Add it to MODEL_PRICING in "
        f"app/infrastructure/llm/pricing.py, or set LLM_PRICING_OVERRIDES."
    )


def describe() -> str:
    from app.infrastructure.llm.client import resolve_provider

    rows = []
    for role in ROLES:
        model = model_for(role.key)
        rows.append((
            role.key,
            role.env,
            model,
            resolve_provider(model).name,
            "yes" if model in MODEL_PRICING else "NO",
            "" if os.getenv(role.env) else f"(default: {DEFAULT_MODEL_ENV})",
        ))

    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    header = ("role", "env var", "model", "provider", "priced")
    widths = [max(w, len(h)) for w, h in zip(widths, header)]

    def line(cells, tail=""):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)) + ("  " + tail if tail else "")

    out = [line(header), line(tuple("-" * w for w in widths))]
    out += [line(r[:5], r[5]) for r in rows]
    return "\n".join(out)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    print(describe())
