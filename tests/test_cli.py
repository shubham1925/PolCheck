from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from polcheck.cli import EXIT_USAGE, main
from polcheck.store import Store
from tests.paths import PICK_SUITE, TABULAR


def run_cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["polcheck", *args])
    with pytest.raises(SystemExit) as exc:
        main()
    code = exc.value.code
    assert isinstance(code, int)
    return code


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "polcheck.toml").write_text(
        'store = ".polcheck"\n[readers.tabular]\nenv_seed = "seed"\nfailure_reason = "failure"\n'
    )
    shutil.copy(PICK_SUITE, tmp_path / "pick.toml")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def ingest_args(path: Path, suite: str = "pick.toml") -> list[str]:
    return ["ingest", str(path), "--reader", "tabular", "--policy", "pick-v3", "--suite", suite]


def test_ingest_csv(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_cli(monkeypatch, *ingest_args(TABULAR / "pick.csv")) == 0
    assert "Ingested 8 runs" in capsys.readouterr().out
    (batch,) = Store(project / ".polcheck").list_batches()
    assert batch.policy_version == "pick-v3"
    assert batch.n_runs == 8


def test_ingest_by_suite_ref_after_registration(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run_cli(monkeypatch, *ingest_args(TABULAR / "pick.csv")) == 0
    assert run_cli(monkeypatch, *ingest_args(TABULAR / "pick.jsonl", suite="pick@1")) == 0
    assert len(Store(project / ".polcheck").list_batches()) == 2


def test_ingest_simulator_flags(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = [
        *ingest_args(TABULAR / "pick.csv"),
        "--simulator-name",
        "isaac-sim",
        "--simulator-version",
        "5.0",
        "--physics-hash",
        "p9",
    ]
    assert run_cli(monkeypatch, *args) == 0
    store = Store(project / ".polcheck")
    (batch,) = store.list_batches()
    run = store.read_batch(batch.batch_id).runs[0]
    assert (run.simulator.name, run.simulator.version, run.physics_hash) == (
        "isaac-sim",
        "5.0",
        "p9",
    )


def test_ingest_unknown_family_exits_4(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_cli(monkeypatch, *ingest_args(TABULAR / "unknown_family.csv")) == EXIT_USAGE
    assert "far-up, middle" in capsys.readouterr().err
    assert Store(project / ".polcheck").list_batches() == []


def test_unregistered_suite_ref_exits_4(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_cli(monkeypatch, *ingest_args(TABULAR / "pick.csv", suite="pick@1")) == EXIT_USAGE
    assert "not registered" in capsys.readouterr().err


def test_missing_option_exits_4_not_2(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_cli(monkeypatch, "ingest", str(TABULAR / "pick.csv")) == EXIT_USAGE
    assert "Missing option" in capsys.readouterr().err


def test_unknown_reader_exits_4(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = ingest_args(TABULAR / "pick.csv")
    args[args.index("tabular")] = "arena"
    assert run_cli(monkeypatch, *args) == EXIT_USAGE


def test_bad_config_exits_4(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (project / "polcheck.toml").write_text("alpha = 7\n")
    assert run_cli(monkeypatch, "reindex") == EXIT_USAGE


def test_internal_error_exits_4(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(self: Store) -> int:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(Store, "reindex", boom)
    assert run_cli(monkeypatch, "reindex") == EXIT_USAGE
    assert "internal error" in capsys.readouterr().err


def test_reindex(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_cli(monkeypatch, *ingest_args(TABULAR / "pick.csv")) == 0
    (project / ".polcheck" / "index.duckdb").unlink()
    assert run_cli(monkeypatch, "reindex") == 0
    assert "Reindexed 1 batches" in capsys.readouterr().out


def test_help_exits_0(monkeypatch: pytest.MonkeyPatch) -> None:
    assert run_cli(monkeypatch, "--help") == 0
