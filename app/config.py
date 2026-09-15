"""Loading `.env`, and reading the settings the app cannot run without.

One rule for `.env`: entry points call `load_env()` before importing anything
else from `app`, and library modules never load it themselves. Many modules
read their settings at import time (`model_for(...)` in every port,
`LOOP_MAX_TURNS`/`MEMO_DIR` in researcher.py), so the load has to come first.

It used to be thirteen `load_dotenv` calls, some with `override=True`
(checkpointer.py, llm.py, main.py) and some without (researcher.py, cli.py,
models.py). Whether a variable set in the shell or the one in `.env` won
depended on which module happened to be imported first — and with
checkpointer.py imported early, `.env` silently beat every command-line
override. Now nothing overrides: the environment wins, `.env` fills the gaps.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv


class MissingSetting(KeyError):
    """A required setting is unset. Replaces the bare KeyError from
    `os.environ[...]` at import time, which named the variable and nothing
    else — not that `.env` was the fix, nor where the list of variables is.
    Still a KeyError, so existing callers that catch one keep working."""

    def __str__(self) -> str:
        # KeyError renders its argument with repr(), quotes and all.
        return str(self.args[0]) if self.args else ""


def load_env() -> None:
    """Load `.env` into the environment without overriding anything already
    set. Idempotent; safe to call from more than one entry point."""
    load_dotenv(override=False)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise MissingSetting(
            f"{name} is not set, and the app cannot start without it. Copy "
            f".env.example to .env and fill it in, or export {name} — "
            f".env.example documents every variable."
        )
    return value
