from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from polcheck.config import Config, ConfigError, load_config

EXAMPLE = Path(__file__).parents[1] / "polcheck.toml.example"


def test_defaults_without_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_config() == Config()


def test_example_file_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "polcheck.toml"
    shutil.copy(EXAMPLE, path)
    config = load_config(path)
    assert config.store == tmp_path / ".polcheck"
    assert config.measures["success_rate"].role == "gate"
    assert config.readers.tabular.env_seed == "seed"


def test_finds_config_in_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "polcheck.toml").write_text('store = "data"\nmin_n = 30\n')
    monkeypatch.chdir(tmp_path)
    config = load_config()
    assert config.min_n == 30
    assert config.store == tmp_path / "data"


def test_absolute_store_kept(tmp_path: Path) -> None:
    path = tmp_path / "polcheck.toml"
    path.write_text(f"store = {str(tmp_path / 'elsewhere')!r}\n")
    assert load_config(path).store == tmp_path / "elsewhere"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("alpha = [", "invalid TOML"),
        ("alpha = 2.0", "alpha"),
        ("colour = 'red'", "colour"),
        ("[rules]\nphysics_hash = 'ignore'", "physics_hash"),
        ("[measures.sparc]\nthreshold = -1", "threshold"),
    ],
)
def test_invalid_config(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "polcheck.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_explicit_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.toml")
