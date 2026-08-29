"""Tests for role -> model configuration.

The property that matters is the fallback: ten knobs exist, but a project
configured with only LLM_CLAUDE_MODEL must behave exactly as it did when
that was the only knob there was.
"""

import importlib

import pytest

from app.infrastructure.llm.models import (
    DEFAULT_MODEL_ENV,
    ROLES,
    configured_models,
    model_env_vars,
    model_for,
    warn_if_unpriced,
)


@pytest.fixture
def only_the_default(monkeypatch):
    """A project that sets nothing but the project-wide model."""
    for role in ROLES:
        monkeypatch.delenv(role.env, raising=False)
    monkeypatch.setenv(DEFAULT_MODEL_ENV, "claude-haiku-4-5-20251001")


def test_every_role_falls_back_to_the_project_wide_model(only_the_default):
    assert set(configured_models().values()) == {"claude-haiku-4-5-20251001"}


@pytest.mark.parametrize("role", ROLES, ids=[r.key for r in ROLES])
def test_each_role_can_be_moved_on_its_own(monkeypatch, only_the_default, role):
    monkeypatch.setenv(role.env, "deepseek-v4-flash")
    models = configured_models()
    assert models[role.key] == "deepseek-v4-flash"
    others = {k: v for k, v in models.items() if k != role.key}
    # Moving one role must not drag the others with it — except when the role
    # IS the project-wide default, which every other role follows by design.
    if role.env == DEFAULT_MODEL_ENV:
        assert set(others.values()) == {"deepseek-v4-flash"}
    else:
        assert set(others.values()) == {"claude-haiku-4-5-20251001"}


def test_role_env_vars_are_unique():
    envs = [role.env for role in ROLES]
    assert len(envs) == len(set(envs))


def test_unknown_role_is_a_loud_error():
    with pytest.raises(KeyError, match="unknown model role"):
        model_for("no-such-role")


def test_the_default_model_is_required(monkeypatch):
    """No built-in fallback id: guessing one would spend money on a model
    nobody chose."""
    for role in ROLES:
        monkeypatch.delenv(role.env, raising=False)
    with pytest.raises(KeyError):
        model_for("agent")


def test_model_env_vars_covers_every_role():
    assert model_env_vars() == [role.key and role.env for role in ROLES]
    assert len(model_env_vars()) == len(ROLES)


def test_the_battery_records_the_registrys_list_not_a_copy():
    """The hand-maintained copy had drifted — it carried two variables
    nothing read and missed ones that do select a model, so a rerun could
    not be checked against the configuration that produced the original."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_battery", "scripts/run_p9_battery.py"
    )
    battery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(battery)

    for env in model_env_vars():
        assert env in battery.MODEL_ENV_VARS


def test_unpriced_model_warns_and_names_the_budget(capsys):
    warn_if_unpriced("some-unpriced-model", "debate", 0.25)
    out = capsys.readouterr().out
    assert "some-unpriced-model" in out
    assert "$0.25" in out


def test_a_priced_model_says_nothing(capsys):
    warn_if_unpriced("deepseek-v4-flash", "debate", 0.25)
    assert capsys.readouterr().out == ""


def test_describe_reports_provider_and_pricing_per_role(monkeypatch, only_the_default):
    from app.infrastructure.llm.models import describe

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("TRADING_DEBATE_MODEL", "deepseek-v4-flash")
    table = describe()
    assert "anthropic" in table and "deepseek" in table
    for role in ROLES:
        assert role.env in table


def test_ports_bind_the_role_the_registry_says_they_do(monkeypatch):
    """A port reading the wrong role's variable would be invisible — the
    model would still be a valid one, just not the one configured.

    Reloading rebinds module-level constants, so the modules are reloaded
    again under the restored environment before this test returns. Without
    that, every later test in the session would see a port pinned to this
    test's models.
    """
    import app.agent.trading.infrastructure.debate_port as debate
    import app.agent.trading.infrastructure.news_digest_port as news
    import app.agent.trading.infrastructure.risk_port as risk

    monkeypatch.setenv(DEFAULT_MODEL_ENV, "claude-haiku-4-5-20251001")
    monkeypatch.setenv("TRADING_DEBATE_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("TRADING_RISK_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("TRADING_NEWS_DIGEST_MODEL", "claude-sonnet-5")

    try:
        assert importlib.reload(debate).DEBATE_MODEL == "deepseek-v4-pro"
        assert importlib.reload(risk).RISK_MODEL == "deepseek-v4-flash"
        assert importlib.reload(news).NEWS_DIGEST_MODEL == "claude-sonnet-5"
    finally:
        monkeypatch.undo()
        for module in (debate, risk, news):
            importlib.reload(module)
