"""Pydantic models for the polcheck data model (BUILD_PLAN section 6).

plus the comparability result (section 7.4). Test results are added in M4.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
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

Severity = Literal["warn", "block"]

ComparabilityStatus = Literal["comparable", "partial", "not_comparable"]

UNKNOWN = "unknown"
"""Simulator name/version or physics hash that the source did not record."""


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

    @model_validator(mode="after")
    def _no_duplicate_scenarios(self) -> Self:
        seen: set[tuple[str, str]] = set()
        for sc in self.scenarios:
            key = (sc.family, json.dumps(sc.config, sort_keys=True))
            if key in seen:
                raise ValueError(f"family {sc.family!r} lists the same config twice")
            seen.add(key)
        return self

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


class BatchData(_Model):
    """A batch together with its runs. This is what readers return."""

    batch: Batch
    runs: list[Run]

    @model_validator(mode="after")
    def _runs_belong_to_batch(self) -> Self:
        if self.batch.n_runs != len(self.runs):
            raise ValueError(f"batch says n_runs={self.batch.n_runs} but has {len(self.runs)} runs")
        for run in self.runs:
            if (run.batch_id, run.suite_ref, run.policy_version) != (
                self.batch.batch_id,
                self.batch.suite_ref,
                self.batch.policy_version,
            ):
                raise ValueError(
                    f"run {run.run_id} does not match its batch "
                    "(batch_id, suite_ref and policy_version must be equal)"
                )
        return self


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


# --- 7.4 Comparability ------------------------------------------------------


class Issue(_Model):
    rule: str
    """`suite_ref`, `scenario_set`, `simulator_version`, `physics_hash` or `min_n`."""
    severity: Severity
    message: str
    """What differs and what to do about it."""


class FamilyCounts(_Model):
    family: str
    n_base: int
    n_cand: int


class ComparabilityResult(_Model):
    status: ComparabilityStatus
    paired: bool
    pairing_reason: str
    """Why the comparison is or is not paired, in words for the report."""
    matched_keys: int
    """`(scenario_id, env_seed)` keys present in both batches (shared scenarios only)."""
    key_overlap: float = Field(ge=0, le=1)
    """`matched_keys` divided by the number of distinct keys across both batches."""
    base_deterministic: bool | None
    cand_deterministic: bool | None
    shared_scenarios: list[ScenarioId]
    families: list[FamilyCounts]
    """Run counts per family over the shared scenarios, one entry per family."""
    low_n_families: list[str]
    """Families below `min_n` on either side; M4 reports these as INCONCLUSIVE."""
    issues: list[Issue]
