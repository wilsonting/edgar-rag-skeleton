from datetime import datetime

import pytest

import app.agent.researcher as researcher
from app.agent.tools import record_log_line, reset_run_provenance


def setup_function():
    reset_run_provenance()


def test_save_output_writes_session_log_sibling(tmp_path, monkeypatch):
    """The saved report gets a -provenance.md sibling holding the run's
    full session log — terminal trace lines plus untruncated tool
    results — so a suspect figure in the report can be traced to the
    exact turn and tool output that produced it after the process exits."""
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)
    record_log_line("--- turn 1 ---")
    record_log_line(
        "  [tool result] Revenue was $64,896 million. [ACN 10-K 2025 §Item 7]"
    )

    path = researcher._save_output("# Memo", "ACN", "fundamentals")

    sibling = path.with_name(f"{path.stem}-provenance.md")
    assert sibling.exists()
    content = sibling.read_text()
    assert "--- turn 1 ---" in content
    assert "64,896" in content


def test_trace_lines_are_recorded_in_session_log(tmp_path, monkeypatch, capsys):
    """_trace both prints to stderr (live console) and records to the
    session log (saved audit trail) — the same line reaches both."""
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)

    researcher._trace("[agent finished after 3 turns]")
    path = researcher._save_output("# Memo", "ACN", "fundamentals")

    sibling = path.with_name(f"{path.stem}-provenance.md")
    assert "[agent finished after 3 turns]" in sibling.read_text()
    assert "[agent finished after 3 turns]" in capsys.readouterr().err


def test_save_output_skips_session_log_for_technical_mode(tmp_path, monkeypatch):
    """The technical interpreter doesn't use the research tools; at its
    save time the module-global session log still holds the preceding
    fundamentals run's trace, so pairing it with the technical report
    would attach the wrong evidence."""
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)
    record_log_line("Leftover fundamentals trace from the same process.")

    path = researcher._save_output("# Technical report", "ACN", "technical")

    assert not path.with_name(f"{path.stem}-provenance.md").exists()


def test_save_output_skips_session_log_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)

    path = researcher._save_output("# Memo", "ACN", "fundamentals")

    assert not path.with_name(f"{path.stem}-provenance.md").exists()


def test_explicit_provenance_is_written_verbatim(tmp_path, monkeypatch):
    """The trading pipeline supplies its own captured terminal log, which
    must land in the sidecar exactly as given."""
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)
    log = "[news] running for ACN as of 2026-08-22\n[sentiment] aggregating 3 of 9\n"

    path = researcher._save_output("# Sentiment", "ACN", "sentiment", provenance=log)

    assert path.with_name(f"{path.stem}-provenance.md").read_text() == log


def test_explicit_provenance_beats_the_session_log(tmp_path, monkeypatch):
    """Both are available here; the explicit one wins. Otherwise a pipeline
    artifact would be paired with whatever trace the preceding fundamentals
    run happened to leave in the module global."""
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)
    record_log_line("Leftover fundamentals trace from the same process.")

    path = researcher._save_output(
        "# Memo", "ACN", "fundamentals", provenance="the real run log"
    )

    content = path.with_name(f"{path.stem}-provenance.md").read_text()
    assert content == "the real run log"
    assert "Leftover" not in content


def test_pipeline_modes_never_inherit_the_session_log(tmp_path, monkeypatch):
    """Same reasoning as technical mode, extended to the artifacts the
    trading CLI writes: they don't call the research tools, so a non-empty
    session log at their save time belongs to some earlier run."""
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)
    record_log_line("Leftover fundamentals trace from the same process.")

    for mode in ("technical", "sentiment", "decision"):
        path = researcher._save_output("# Report", "ACN", mode)
        assert not path.with_name(f"{path.stem}-provenance.md").exists(), mode


def test_sentiment_and_decision_land_in_the_dated_folder(tmp_path, monkeypatch):
    """Same layout as fundamentals/technical, so one run's artifacts sit
    together in the vault rather than scattering across two levels."""
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)

    sentiment = researcher._save_output("# S", "ACN", "sentiment")
    decision = researcher._save_output("# D", "ACN", "decision")

    assert sentiment.parent == decision.parent
    assert sentiment.parent.parent.name == "ACN"
    assert sentiment.parent.name.isdigit() and len(sentiment.parent.name) == 8
    assert sentiment.name.startswith("ACN-sentiment-")
    assert decision.name.startswith("ACN-decision-")


# ---------------------------------------------------------------------------
# vault_run — one folder per pipeline run
# ---------------------------------------------------------------------------

def test_a_run_folder_gathers_every_artifact_of_one_run(tmp_path, monkeypatch):
    """The layout this exists to produce. Without it a run scatters six files
    into the ticker's dated folder alongside every previous run's six, and
    telling one run's artifacts from another's means reading timestamps."""
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)

    with researcher.vault_run(datetime(2026, 8, 22, 7, 1, 53)) as folder:
        paths = {
            mode: researcher._save_output(f"# {mode}", "ACN", mode)
            for mode in ("fundamentals", "technical", "sentiment", "decision", "debate")
        }

    assert folder == "2026-0822-070153"
    assert len({p.parent for p in paths.values()}) == 1
    run_dir = paths["decision"].parent
    assert run_dir == tmp_path / "ACN" / "20260822" / "2026-0822-070153"
    assert sorted(p.name for p in paths.values()) == [
        "ACN-debate.md",
        "ACN-decision.md",
        "ACN-fundamental.md",
        "ACN-sentiment.md",
        "ACN-technical.md",
    ]


def test_provenance_siblings_land_in_the_same_run_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)
    record_log_line("--- turn 1 ---")

    with researcher.vault_run(datetime(2026, 8, 22, 7, 1, 53)):
        fundamentals = researcher._save_output("# F", "ACN", "fundamentals")
        sentiment = researcher._save_output(
            "# S", "ACN", "sentiment", provenance="the run log"
        )

    assert sorted(p.name for p in fundamentals.parent.iterdir()) == [
        "ACN-fundamental-provenance.md",
        "ACN-fundamental.md",
        "ACN-sentiment-provenance.md",
        "ACN-sentiment.md",
    ]
    assert sentiment.with_name("ACN-sentiment-provenance.md").read_text() == "the run log"


def test_a_run_that_saves_nothing_leaves_no_empty_folder(tmp_path, monkeypatch):
    """The directory is created by the first save, not by entering the block —
    otherwise a run that dies early litters the vault, which is worse than
    the scattering this replaces."""
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)

    with researcher.vault_run():
        pass

    assert list(tmp_path.iterdir()) == []


def test_two_runs_get_two_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)

    with researcher.vault_run(datetime(2026, 8, 22, 7, 1, 53)):
        first = researcher._save_output("# D", "ACN", "decision")
    with researcher.vault_run(datetime(2026, 8, 22, 9, 30, 0)):
        second = researcher._save_output("# D", "ACN", "decision")

    assert first.parent != second.parent
    assert first.parent.parent == second.parent.parent   # same dated folder
    assert first.name == second.name == "ACN-decision.md"


def test_a_second_artifact_of_the_same_mode_refuses_to_overwrite(tmp_path, monkeypatch):
    """Unreachable today — every mode saves exactly once per run — but a
    silent overwrite would destroy a report that cost real money, and the
    filename no longer carries a timestamp to keep them apart."""
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)

    with researcher.vault_run():
        first = researcher._save_output("# first", "ACN", "decision")
        with pytest.raises(FileExistsError, match="would overwrite"):
            researcher._save_output("# second", "ACN", "decision")

    assert first.read_text() == "# first"


def test_outside_a_run_the_flat_timestamped_layout_is_unchanged(tmp_path, monkeypatch):
    """The standalone research CLI writes one report per invocation, so it
    has nothing to gather and keeps the layout its existing notes use."""
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)

    path = researcher._save_output("# Memo", "ACN", "fundamentals")

    assert path.parent == tmp_path / "ACN" / datetime.now().strftime("%Y%m%d")
    assert path.name.startswith("ACN-fundamental-")
    assert path.name.endswith(".md")


def test_the_date_folder_comes_from_the_run_start_not_the_save_time(tmp_path, monkeypatch):
    """A run that starts at 23:58 and finishes after midnight must not file
    its fundamentals under one date and its debate transcript under the
    next — that is the same scattering, harder to spot."""
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)

    with researcher.vault_run(datetime(2026, 8, 22, 23, 58, 0)):
        path = researcher._save_output("# D", "ACN", "decision")

    assert path.parent == tmp_path / "ACN" / "20260822" / "2026-0822-235800"


def test_the_run_folder_is_restored_on_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)
    assert researcher._RUN_STAMP is None

    with researcher.vault_run():
        assert researcher._RUN_STAMP is not None

    assert researcher._RUN_STAMP is None
