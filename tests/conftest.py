"""Shared test setup.

The provider layer validates credentials at CLIENT CONSTRUCTION rather than
at the first API call, so a missing key surfaces as a config error before
any money is spent instead of as a 401 thirty turns into a run. That is the
right trade in production and the wrong one in a suite that builds clients
it never calls: several port tests stub the call layer (`_summarize_batch`,
`create_with_temperature_fallback`) but still run the line that constructs
the client.

So: a placeholder key for every non-Anthropic provider, set only when the
real environment has none. Never overwrites a key that is actually present
— a test that DOES reach the network keeps whatever the developer
configured.
"""

import os

import pytest

from app.config import load_env

# The suite imports app modules that read settings at import time, and those
# modules no longer load .env themselves (app/config.py) — so it is loaded
# here, the same way an entry point would, without overriding anything the
# environment already set. CI has no .env and sets what it needs in the
# workflow; locally this keeps the database URLs and required settings a
# developer's .env provides.
load_env()

_PLACEHOLDER_KEY_ENVS = ("DEEPSEEK_API_KEY", "LLM_API_KEY")


@pytest.fixture(autouse=True)
def _placeholder_provider_keys(monkeypatch):
    for name in _PLACEHOLDER_KEY_ENVS:
        if not os.getenv(name):
            monkeypatch.setenv(name, "test-placeholder-not-a-real-key")
