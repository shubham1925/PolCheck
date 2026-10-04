"""Pydantic models for the polcheck data model (BUILD_PLAN section 6).

Comparison-side models (comparability and test results) are added with the
milestones that produce them (M2, M4).
"""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
)

SuiteName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
"""Suite names may not contain `@`, so `name@version` parses unambiguously."""

SuiteRef = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*@[1-9][0-9]*$")]
"""A suite reference, `name@version`."""

ScenarioId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{16}$")]
"""First 16 hex chars of the SHA-256 scenario fingerprint (section 6.1)."""

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
"""Metric values must be finite. An undefined measure is absent, never NaN."""

RunSource = Literal["native", "tabular", "arena"]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- 6.1 Suite and scenario -------------------------------------------------


class SeedRange(_Model):
    start: int = Field(ge=0)
    count: int = Field(gt=0)

    def seeds(self) -> range:
        return range(self.start, self.start + self.count)


class Scenario(_Model):
    family: NonEmptyStr
    config: dict[str, JsonValue] = Field(default_factory=dict)
    """Opaque to the core. Only readers and measures may interpret it."""
    seeds: SeedRange


class Suite(_Model):
    name: SuiteName
    version: int = Field(ge=1)
    scenarios: list[Scenario] = Field(min_length=1)

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"


# --- 6.2 Run ----------------------------------------------------------------


class SimulatorInfo(_Model):
    name: NonEmptyStr
    version: NonEmptyStr


class Run(_Model):
    run_id: UUID
    batch_id: UUID
    policy_version: NonEmptyStr
    suite_ref: SuiteRef
    scenario_id: ScenarioId
    family: NonEmptyStr
    env_seed: int
    policy_seed: int | None = None
    simulator: SimulatorInfo
    physics_hash: NonEmptyStr
    success: bool
    failure_reason: str | None = None
    metrics: dict[str, FiniteFloat] = Field(default_factory=dict)
    source: RunSource
    has_timeseries: bool
    started_at: AwareDatetime


# --- 6.3 Batch --------------------------------------------------------------


class Batch(_Model):
    batch_id: UUID
    policy_version: NonEmptyStr
    suite_ref: SuiteRef
    created_at: AwareDatetime
    git_sha: str | None = None
    deterministic: bool | None = None
    """None means unknown; see the determinism helper (section 7.4)."""
    n_runs: int = Field(ge=0)


# --- 6.6 Portable batch file ------------------------------------------------


class ManifestContents(_Model):
    timeseries: bool = False
    contacts: bool = False


class BatchManifest(_Model):
    """`manifest.json` inside a `.polcheck` export file."""

    format_version: Literal[1] = 1
    batch: Batch
    suite_toml: str
    """The suite file the batch was recorded against, verbatim."""
    measure_versions: dict[str, str] = Field(default_factory=dict)
    contents: ManifestContents = Field(default_factory=ManifestContents)
    is_baseline: bool = False
    """True if the batch was its suite's baseline at export time."""
