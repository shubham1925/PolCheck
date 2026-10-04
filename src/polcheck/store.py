"""Parquet + DuckDB storage (BUILD_PLAN 6.5).

Files are the source of truth; `index.duckdb` is a cache that `reindex()`
rebuilds from them. Layout under the store root:

    index.duckdb
    baselines.json                      # current baselines + frozen suite hashes
    suites/<name>@<version>.toml        # copy of every suite used by a batch
    batches/<batch_id>/batch.json       # batch row; written last, marks the batch complete
    batches/<batch_id>/runs.parquet     # summary rows, metrics as `metric.<name>` columns
    batches/<batch_id>/ts/<run_id>.parquet
    batches/<batch_id>/contacts/<run_id>.parquet
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from polcheck.schema import Batch, BatchData, Run, SimulatorInfo, Suite
from polcheck.suite import parse_suite, suite_hash

INDEX_VERSION = 1
METRIC_PREFIX = "metric."

RUN_COLUMNS = pa.schema(
    [
        ("run_id", pa.string()),
        ("batch_id", pa.string()),
        ("policy_version", pa.string()),
        ("suite_ref", pa.string()),
        ("scenario_id", pa.string()),
        ("family", pa.string()),
        ("env_seed", pa.int64()),
        ("policy_seed", pa.int64()),
        ("simulator_name", pa.string()),
        ("simulator_version", pa.string()),
        ("physics_hash", pa.string()),
        ("success", pa.bool_()),
        ("failure_reason", pa.string()),
        ("source", pa.string()),
        ("has_timeseries", pa.bool_()),
        ("started_at", pa.timestamp("us", tz="UTC")),
    ]
)

CONTACT_COLUMNS = pa.schema(
    [
        ("t", pa.float64()),
        ("geom_a", pa.string()),
        ("geom_b", pa.string()),
        ("force_norm", pa.float64()),
        ("intended", pa.bool_()),
    ]
)

_INDEX_SCHEMA = """
CREATE TABLE meta (key VARCHAR PRIMARY KEY, value VARCHAR);
CREATE TABLE batches (
    batch_id VARCHAR PRIMARY KEY,
    policy_version VARCHAR NOT NULL,
    suite_ref VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL,  -- UTC
    git_sha VARCHAR,
    deterministic BOOLEAN,
    n_runs INTEGER NOT NULL
);
CREATE TABLE runs (
    run_id VARCHAR PRIMARY KEY,
    batch_id VARCHAR NOT NULL,
    policy_version VARCHAR NOT NULL,
    suite_ref VARCHAR NOT NULL,
    scenario_id VARCHAR NOT NULL,
    family VARCHAR NOT NULL,
    env_seed BIGINT NOT NULL,
    policy_seed BIGINT,
    simulator_name VARCHAR NOT NULL,
    simulator_version VARCHAR NOT NULL,
    physics_hash VARCHAR NOT NULL,
    success BOOLEAN NOT NULL,
    failure_reason VARCHAR,
    source VARCHAR NOT NULL,
    has_timeseries BOOLEAN NOT NULL,
    started_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE baselines (suite_ref VARCHAR PRIMARY KEY, batch_id VARCHAR NOT NULL);
"""


class StoreError(Exception):
    pass


class SuiteFrozenError(StoreError):
    pass


class _BaselineState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baselines: dict[str, UUID] = Field(default_factory=dict)
    """suite_ref -> current baseline batch."""
    frozen: dict[str, str] = Field(default_factory=dict)
    """suite_ref -> suite_hash at the time it first became a baseline."""


# --- Parquet conversion -----------------------------------------------------


def runs_to_table(runs: Sequence[Run]) -> pa.Table:
    metric_names = sorted({name for run in runs for name in run.metrics})
    columns: dict[str, pa.Array] = {
        "run_id": pa.array([str(r.run_id) for r in runs], pa.string()),
        "batch_id": pa.array([str(r.batch_id) for r in runs], pa.string()),
        "policy_version": pa.array([r.policy_version for r in runs], pa.string()),
        "suite_ref": pa.array([r.suite_ref for r in runs], pa.string()),
        "scenario_id": pa.array([r.scenario_id for r in runs], pa.string()),
        "family": pa.array([r.family for r in runs], pa.string()),
        "env_seed": pa.array([r.env_seed for r in runs], pa.int64()),
        "policy_seed": pa.array([r.policy_seed for r in runs], pa.int64()),
        "simulator_name": pa.array([r.simulator.name for r in runs], pa.string()),
        "simulator_version": pa.array([r.simulator.version for r in runs], pa.string()),
        "physics_hash": pa.array([r.physics_hash for r in runs], pa.string()),
        "success": pa.array([r.success for r in runs], pa.bool_()),
        "failure_reason": pa.array([r.failure_reason for r in runs], pa.string()),
        "source": pa.array([r.source for r in runs], pa.string()),
        "has_timeseries": pa.array([r.has_timeseries for r in runs], pa.bool_()),
        "started_at": pa.array([r.started_at for r in runs], RUN_COLUMNS.field("started_at").type),
    }
    for name in metric_names:
        columns[METRIC_PREFIX + name] = pa.array([r.metrics.get(name) for r in runs], pa.float64())
    return pa.table(columns)


def table_to_runs(table: pa.Table) -> list[Run]:
    runs = []
    for row in table.to_pylist():
        metrics = {
            key[len(METRIC_PREFIX) :]: value
            for key, value in row.items()
            if key.startswith(METRIC_PREFIX) and value is not None
        }
        runs.append(
            Run(
                run_id=UUID(row["run_id"]),
                batch_id=UUID(row["batch_id"]),
                policy_version=row["policy_version"],
                suite_ref=row["suite_ref"],
                scenario_id=row["scenario_id"],
                family=row["family"],
                env_seed=row["env_seed"],
                policy_seed=row["policy_seed"],
                simulator=SimulatorInfo(
                    name=row["simulator_name"], version=row["simulator_version"]
                ),
                physics_hash=row["physics_hash"],
                success=row["success"],
                failure_reason=row["failure_reason"],
                metrics=metrics,
                source=row["source"],
                has_timeseries=row["has_timeseries"],
                started_at=row["started_at"],
            )
        )
    return runs


def read_batch_dir(path: Path) -> BatchData:
    """Read a complete batch directory (one with `batch.json`)."""
    batch_file = path / "batch.json"
    if not batch_file.is_file():
        raise StoreError(f"{path} is not a complete batch (no batch.json)")
    batch = Batch.model_validate_json(batch_file.read_text(encoding="utf-8"))
    runs = table_to_runs(pq.read_table(path / "runs.parquet"))
    return BatchData(batch=batch, runs=runs)


# --- File helpers -----------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _to_utc_naive(dt: datetime) -> datetime:
    return dt.astimezone(UTC).replace(tzinfo=None)


# --- Store ------------------------------------------------------------------


class Store:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def index_path(self) -> Path:
        return self.root / "index.duckdb"

    @property
    def _baselines_path(self) -> Path:
        return self.root / "baselines.json"

    def batch_dir(self, batch_id: UUID) -> Path:
        return self.root / "batches" / str(batch_id)

    def _suite_path(self, ref: str) -> Path:
        return self.root / "suites" / f"{ref}.toml"

    # suites

    def register_suite(self, path: Path) -> Suite:
        """Copy a suite TOML into the store, unless an identical one is there.

        Replacing a suite version that has been a baseline is refused.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise StoreError(f"suite file not found: {path}") from exc
        suite = parse_suite(text, source=str(path))
        target = self._suite_path(suite.ref)
        if target.is_file():
            existing = parse_suite(target.read_text(encoding="utf-8"), source=str(target))
            if suite_hash(existing) == suite_hash(suite):
                return existing
            frozen_hash = self._read_baselines().frozen.get(suite.ref)
            if frozen_hash is not None and frozen_hash != suite_hash(suite):
                raise SuiteFrozenError(
                    f"suite {suite.ref} is frozen because one of its batches has been a "
                    f"baseline, and {path} changes its scenarios. Bump the version in "
                    "[suite] to record the new scenarios."
                )
        _atomic_write_text(target, text)
        return suite

    def get_suite(self, ref: str) -> Suite:
        return parse_suite(self.suite_toml(ref), source=str(self._suite_path(ref)))

    def suite_toml(self, ref: str) -> str:
        path = self._suite_path(ref)
        if not path.is_file():
            raise StoreError(
                f"suite {ref} is not registered in {self.root}; pass the path to its "
                "TOML file instead to register it"
            )
        return path.read_text(encoding="utf-8")

    def resolve_suite(self, arg: str | Path) -> Suite:
        """`arg` is a path to a suite TOML (registered on the way) or a `name@version`."""
        path = Path(arg)
        if path.suffix == ".toml" or path.is_file():
            return self.register_suite(path)
        return self.get_suite(str(arg))

    # baselines

    def _read_baselines(self) -> _BaselineState:
        if not self._baselines_path.is_file():
            return _BaselineState()
        return _BaselineState.model_validate_json(self._baselines_path.read_text(encoding="utf-8"))

    def set_baseline(self, batch_id: UUID) -> None:
        """Make a batch its suite's baseline and freeze that suite version."""
        batch = self.read_batch_meta(batch_id)
        suite = self.get_suite(batch.suite_ref)
        state = self._read_baselines()
        state.baselines[batch.suite_ref] = batch_id
        state.frozen.setdefault(batch.suite_ref, suite_hash(suite))
        _atomic_write_text(self._baselines_path, state.model_dump_json(indent=2))
        with self._index() as con:
            con.execute("DELETE FROM baselines WHERE suite_ref = ?", [batch.suite_ref])
            con.execute("INSERT INTO baselines VALUES (?, ?)", [batch.suite_ref, str(batch_id)])

    def baseline(self, suite_ref: str) -> UUID | None:
        return self._read_baselines().baselines.get(suite_ref)

    def is_frozen(self, suite_ref: str) -> bool:
        return suite_ref in self._read_baselines().frozen

    # batches

    def write_timeseries(self, batch_id: UUID, run_id: UUID, table: pa.Table) -> None:
        _atomic_write_parquet(self.batch_dir(batch_id) / "ts" / f"{run_id}.parquet", table)

    def write_contacts(self, batch_id: UUID, run_id: UUID, table: pa.Table) -> None:
        _atomic_write_parquet(
            self.batch_dir(batch_id) / "contacts" / f"{run_id}.parquet",
            table.cast(CONTACT_COLUMNS),
        )

    def write_batch(self, data: BatchData) -> None:
        """Write a batch's summary and index it. Time-series are written beforehand."""
        batch = data.batch
        directory = self.batch_dir(batch.batch_id)
        if (directory / "batch.json").exists():
            raise StoreError(f"batch {batch.batch_id} already exists in {self.root}")
        self.suite_toml(batch.suite_ref)  # the suite must be registered
        _atomic_write_parquet(directory / "runs.parquet", runs_to_table(data.runs))
        _atomic_write_text(directory / "batch.json", batch.model_dump_json(indent=2))
        with self._index() as con:
            exists = con.execute(
                "SELECT 1 FROM batches WHERE batch_id = ?", [str(batch.batch_id)]
            ).fetchone()
            if exists is None:  # a rebuild triggered by _index() may have added it already
                _index_batch(con, directory, batch)

    def read_batch(self, batch_id: UUID) -> BatchData:
        return read_batch_dir(self.batch_dir(batch_id))

    def read_batch_meta(self, batch_id: UUID) -> Batch:
        path = self.batch_dir(batch_id) / "batch.json"
        if not path.is_file():
            raise StoreError(f"no batch {batch_id} in {self.root}")
        return Batch.model_validate_json(path.read_text(encoding="utf-8"))

    def read_timeseries(self, batch_id: UUID, run_id: UUID) -> pa.Table | None:
        path = self.batch_dir(batch_id) / "ts" / f"{run_id}.parquet"
        return pq.read_table(path) if path.is_file() else None

    def read_contacts(self, batch_id: UUID, run_id: UUID) -> pa.Table | None:
        path = self.batch_dir(batch_id) / "contacts" / f"{run_id}.parquet"
        return pq.read_table(path) if path.is_file() else None

    def _complete_batch_dirs(self) -> list[Path]:
        root = self.root / "batches"
        if not root.is_dir():
            return []
        return sorted(p for p in root.iterdir() if (p / "batch.json").is_file())

    # index

    def reindex(self) -> int:
        """Rebuild `index.duckdb` from files. Returns the number of batches indexed."""
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_name(self.index_path.name + ".tmp")
        for leftover in (tmp, tmp.with_name(tmp.name + ".wal")):
            leftover.unlink(missing_ok=True)
        dirs = self._complete_batch_dirs()
        con = duckdb.connect(str(tmp))
        try:
            con.execute(_INDEX_SCHEMA)
            con.execute("INSERT INTO meta VALUES ('index_version', ?)", [str(INDEX_VERSION)])
            for directory in dirs:
                batch = Batch.model_validate_json(
                    (directory / "batch.json").read_text(encoding="utf-8")
                )
                _index_batch(con, directory, batch)
            for suite_ref, batch_id in self._read_baselines().baselines.items():
                con.execute("INSERT INTO baselines VALUES (?, ?)", [suite_ref, str(batch_id)])
        finally:
            con.close()
        os.replace(tmp, self.index_path)
        return len(dirs)

    def _index_is_current(self) -> bool:
        if not self.index_path.is_file():
            return False
        try:
            con = duckdb.connect(str(self.index_path), read_only=True)
        except duckdb.Error:
            return False
        try:
            row = con.execute("SELECT value FROM meta WHERE key = 'index_version'").fetchone()
        except duckdb.Error:
            return False
        finally:
            con.close()
        return row is not None and row[0] == str(INDEX_VERSION)

    @contextmanager
    def _index(self) -> Iterator[duckdb.DuckDBPyConnection]:
        if not self._index_is_current():
            self.reindex()
        con = duckdb.connect(str(self.index_path))
        try:
            yield con
        finally:
            con.close()

    def query(self, sql: str, params: Sequence[Any] | None = None) -> list[tuple[Any, ...]]:
        """Run a read query against the index (rebuilding it first if missing)."""
        with self._index() as con:
            return con.execute(sql, params).fetchall()

    def list_batches(
        self, *, policy_version: str | None = None, suite_ref: str | None = None
    ) -> list[Batch]:
        rows = self.query(
            """
            SELECT batch_id, policy_version, suite_ref, created_at, git_sha, deterministic, n_runs
            FROM batches
            WHERE (? IS NULL OR policy_version = ?) AND (? IS NULL OR suite_ref = ?)
            ORDER BY created_at, batch_id
            """,
            [policy_version, policy_version, suite_ref, suite_ref],
        )
        return [
            Batch(
                batch_id=UUID(r[0]),
                policy_version=r[1],
                suite_ref=r[2],
                created_at=r[3].replace(tzinfo=UTC),
                git_sha=r[4],
                deterministic=r[5],
                n_runs=r[6],
            )
            for r in rows
        ]

    def latest_batch(self, policy_version: str, suite_ref: str | None = None) -> Batch | None:
        batches = self.list_batches(policy_version=policy_version, suite_ref=suite_ref)
        return batches[-1] if batches else None


def _index_batch(con: duckdb.DuckDBPyConnection, directory: Path, batch: Batch) -> None:
    con.execute(
        "INSERT INTO batches VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            str(batch.batch_id),
            batch.policy_version,
            batch.suite_ref,
            _to_utc_naive(batch.created_at),
            batch.git_sha,
            batch.deterministic,
            batch.n_runs,
        ],
    )
    runs_path = directory / "runs.parquet"
    for name in pq.read_schema(runs_path).names:
        if name.startswith(METRIC_PREFIX):
            con.execute(f"ALTER TABLE runs ADD COLUMN IF NOT EXISTS {_sql_ident(name)} DOUBLE")
    con.execute(
        f"INSERT INTO runs BY NAME SELECT * FROM read_parquet({_sql_str(runs_path.as_posix())})"
    )
