"""Loads `polcheck.toml` (BUILD_PLAN 7.9)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from polcheck.schema import SimulatorInfo

CONFIG_FILENAME = "polcheck.toml"

Severity = Literal["warn", "block"]


class ConfigError(ValueError):
    pass


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RulesConfig(_Model):
    suite_ref: Severity = "block"
    scenario_set: Severity = "warn"
    simulator_version: Severity = "warn"
    physics_hash: Severity = "block"


class MeasureConfig(_Model):
    worse: Literal["higher", "lower"] | None = None
    threshold: float | None = Field(default=None, gt=0)
    role: Literal["gate", "advisory"] | None = None


class TabularMapping(_Model):
    """Column mapping for the `tabular` reader. Values are column names."""

    family: str = "family"
    env_seed: str = "env_seed"
    success: str = "success"
    policy_seed: str | None = None
    failure_reason: str | None = None
    scenario_index: str | None = None
    metrics: list[str] | None = None
    """Metric columns to import. None means every other numeric column."""
    simulator: SimulatorInfo | None = None
    physics_hash: str | None = None


class ReadersConfig(_Model):
    tabular: TabularMapping = Field(default_factory=TabularMapping)


class Config(_Model):
    store: Path = Path(".polcheck")
    alpha: float = Field(default=0.05, gt=0, lt=1)
    fdr_q: float = Field(default=0.05, gt=0, lt=1)
    min_n: int = Field(default=20, ge=1)
    min_n_tail: int = Field(default=100, ge=1)
    bootstrap_resamples: int = Field(default=10_000, ge=100)
    report_max_run_pages: int = Field(default=50, ge=0)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    measures: dict[str, MeasureConfig] = Field(default_factory=dict)
    readers: ReadersConfig = Field(default_factory=ReadersConfig)


def load_config(path: Path | None = None) -> Config:
    """Load config from `path`, or from `./polcheck.toml` if it exists, else defaults.

    A relative `store` is resolved against the config file's directory, so the
    same file works from any working directory.
    """
    if path is None:
        candidate = Path.cwd() / CONFIG_FILENAME
        if not candidate.is_file():
            return Config()
        path = candidate
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    try:
        config = Config.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if not config.store.is_absolute():
        config = config.model_copy(update={"store": path.parent / config.store})
    return config
