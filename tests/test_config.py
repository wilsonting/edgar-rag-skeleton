"""Configuration: one way `.env` is loaded, clear errors for what is required.

docs/code_review.md, Medium #12: thirteen `load_dotenv` calls, some with
`override=True` in library modules, made shell-vs-.env precedence depend on
import order; required settings failed as a bare KeyError at import time;
EMBEDDING_MODEL and OPENAI_MODEL were configurable in name only.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import app.config as config
from app.application.embedding_service import EmbeddingService
from app.config import MissingSetting, require_env

APP = Path(__file__).resolve().parents[1] / "app"

def test_load_env_never_overrides_the_environment(monkeypatch):
    calls = []
    monkeypatch.setattr(config, "load_dotenv", lambda **kw: calls.append(kw))
    config.load_env()
    assert calls == [{"override": False}]


def test_a_missing_required_setting_says_what_to_do(monkeypatch):
    monkeypatch.delenv("LOOP_MAX_TURNS", raising=False)
    with pytest.raises(MissingSetting) as exc:
        require_env("LOOP_MAX_TURNS")
    message = str(exc.value)
    assert message.startswith("LOOP_MAX_TURNS is not set")
    assert ".env.example" in message
    assert isinstance(exc.value, KeyError)   # existing `except KeyError` still works


def test_a_set_setting_is_returned(monkeypatch):
    monkeypatch.setenv("LOOP_MAX_TURNS", "45")
    assert require_env("LOOP_MAX_TURNS") == "45"


def test_only_app_config_loads_dotenv():
    """Library modules loading .env — with override=True, in checkpointer.py
    and llm.py — is what made precedence depend on import order.

    Covers all of `app/` now. It used to exempt ingest.py and db_postgres.py,
    the last two modules still calling `load_dotenv(override=True)` — both
    part of the pre-refactor PDF pipeline, now deleted. An exemption that
    exists only for dead code is a reason to delete the code.
    """
    offenders = []
    for path in APP.rglob("*.py"):
        if path == APP / "config.py":
            continue
        source = path.read_text()
        if re.search(r"\bload_dotenv\b", source) or "override=True" in source:
            offenders.append(str(path.relative_to(APP)))
    assert offenders == []


@pytest.mark.parametrize("module", [
    "main.py", "cli.py", "agent/researcher.py", "agent/trading/interface/cli.py",
])
def test_entry_points_load_env_before_any_other_app_import(module):
    """Settings are read at import time all over the app (every port's
    model_for), so the load must come before the first other app import."""
    tree = ast.parse((APP / module).read_text())
    first_app_import = None
    load_line = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("app"):
            if node.module == "app.config":
                continue
            first_app_import = min(first_app_import or node.lineno, node.lineno)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "load_env"
        ):
            load_line = node.lineno
    assert load_line is not None, f"{module} never calls load_env()"
    assert load_line < first_app_import


def test_embedding_model_setting_takes_effect(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    assert EmbeddingService().model == "text-embedding-ada-002"


def test_embedding_model_defaults_to_the_one_the_schema_is_sized_for(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    assert EmbeddingService().model == "text-embedding-3-small"


def test_openai_model_is_gone_from_the_example_env():
    """Read by nothing; it was only recorded into battery manifests, where it
    looked like configuration that had shaped the run."""
    example = (APP.parent / ".env.example").read_text()
    assert "OPENAI_MODEL" not in example


def test_every_required_setting_is_read_through_require_env():
    """`os.environ["X"]` raises a bare KeyError naming the variable and
    nothing else — not that `.env` is the fix, nor where the list of
    variables lives. That is what `require_env` exists to replace, and six
    reads still bypassed it: EDGAR_USER_AGENT in four places,
    POSTGRES_DATABASE_URL in the pool, and TRADING_CHECKPOINT_DB_URI, which
    was an `os.getenv` whose None reached `AsyncConnectionPool(conninfo=None)`
    and failed several frames away naming nothing."""
    offenders = []
    for path in APP.rglob("*.py"):
        if path == APP / "config.py":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if re.search(r"os\.environ\[", line):
                offenders.append(f"{path.relative_to(APP)}:{lineno}")
    assert offenders == []


def test_the_dead_pdf_pipeline_is_gone():
    """`ingest.py`, `retrieve.py`, `db_postgres.py`, `db.py` and `chunk.py`
    were the pre-refactor tutorial pipeline. Nothing imported them, and
    `ingest.py` wrote to a `chunks(source, ...)` schema that no longer
    exists. Their only remaining effect was to force an exemption in
    `test_only_app_config_loads_dotenv`."""
    for name in ("ingest.py", "retrieve.py", "db_postgres.py", "db.py", "chunk.py"):
        assert not (APP / name).exists(), name


def test_every_variable_the_code_reads_is_in_the_example_env():
    """`.env.example` is what `require_env`'s error message points at, so a
    variable missing from it is a dead end for the person reading that
    error. AGENT_TOOL_CONCURRENCY was read and documented nowhere."""
    source = "\n".join(p.read_text() for p in APP.rglob("*.py"))
    read = {
        name
        for tup in re.findall(
            r'os\.getenv\("([A-Z_0-9]{3,})"|require_env\("([A-Z_0-9]{3,})"', source
        )
        for name in tup
        if name
    }
    documented = set(re.findall(
        r"^#?\s*([A-Z_0-9]{3,})=", (APP.parent / ".env.example").read_text(), re.M
    ))
    assert sorted(read - documented) == []
