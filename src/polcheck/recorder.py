"""Native recorder (BUILD_PLAN 7.1)."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import NamedTuple, Self
from uuid import UUID, uuid4

import numpy as np
import pyarrow as pa
from numpy.typing import ArrayLike

from polcheck.schema import Batch, BatchData, Run, Scenario, SimulatorInfo
from polcheck.store import CONTACT_COLUMNS, Store
from polcheck.suite import scenario_id

log = logging.getLogger(__name__)

UNKNOWN_SIMULATOR = SimulatorInfo(name="unknown", version="unknown")


class RecorderError(RuntimeError):
    """Misuse of the recorder API. Never swallowed as a run failure."""


class Contact(NamedTuple):
    geom_a: str
    geom_b: str
    force_norm: float
    intended: bool


def _vector(name: str, value: ArrayLike, length: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise RecorderError(f"{name} must be a 1-D array, got shape {arr.shape}")
    if length is not None and arr.shape[0] != length:
        raise RecorderError(f"{name} must have {length} elements, got {arr.shape[0]}")
    return arr


def flatten_step(
    *,
    joint_pos: ArrayLike | None = None,
    joint_vel: ArrayLike | None = None,
    joint_torque: ArrayLike | None = None,
    ee_pos: ArrayLike | None = None,
    ee_quat: ArrayLike | None = None,
    objects: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
    clearance: float | None = None,
    qpos: ArrayLike | None = None,
) -> dict[str, float]:
    """Flatten one step's signals into the column names of section 6.4."""
    row: dict[str, float] = {}
    for prefix, joint_values in (
        ("joint_pos", joint_pos),
        ("joint_vel", joint_vel),
        ("joint_torque", joint_torque),
    ):
        if joint_values is not None:
            for i, v in enumerate(_vector(prefix, joint_values)):
                row[f"{prefix}.{i}"] = float(v)
    if ee_pos is not None:
        row.update(
            zip(
                ("ee_pos.x", "ee_pos.y", "ee_pos.z"),
                _vector("ee_pos", ee_pos, 3).tolist(),
                strict=True,
            )
        )
    if ee_quat is not None:
        quat = _vector("ee_quat", ee_quat, 4).tolist()
        row.update(zip(("ee_quat.w", "ee_quat.x", "ee_quat.y", "ee_quat.z"), quat, strict=True))
    for obj, (pos, quat_) in (objects or {}).items():
        p = _vector(f"objects[{obj!r}] position", pos, 3).tolist()
        q = _vector(f"objects[{obj!r}] quaternion", quat_, 4).tolist()
        row.update(zip((f"obj.{obj}.pos.{a}" for a in "xyz"), p, strict=True))
        row.update(zip((f"obj.{obj}.quat.{a}" for a in "wxyz"), q, strict=True))
    if clearance is not None:
        row["clearance"] = float(clearance)
    if qpos is not None:
        for i, v in enumerate(_vector("qpos", qpos)):
            row[f"qpos.{i}"] = float(v)
    return row


class RunContext:
    """One run being recorded. Created by `Recorder.run`."""

    def __init__(
        self,
        recorder: Recorder,
        scenario_id: str,
        family: str,
        env_seed: int,
        policy_seed: int | None,
    ) -> None:
        self._recorder = recorder
        self.run_id = uuid4()
        self.scenario_id = scenario_id
        self.family = family
        self.env_seed = env_seed
        self.policy_seed = policy_seed
        self.started_at = datetime.now(UTC)
        self.finished = False
        self._columns: dict[str, list[float]] | None = None
        self._contacts: list[tuple[float, Contact]] = []

    def log(
        self,
        *,
        t: float,
        joint_pos: ArrayLike | None = None,
        joint_vel: ArrayLike | None = None,
        joint_torque: ArrayLike | None = None,
        ee_pos: ArrayLike | None = None,
        ee_quat: ArrayLike | None = None,
        objects: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
        clearance: float | None = None,
        qpos: ArrayLike | None = None,
    ) -> None:
        """Buffer one control step. Every step of a run must log the same signals."""
        self._check_open()
        row = {"t": float(t)} | flatten_step(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            joint_torque=joint_torque,
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            objects=objects,
            clearance=clearance,
            qpos=qpos,
        )
        if self._columns is None:
            self._columns = {key: [] for key in row}
        elif row.keys() != self._columns.keys():
            added = sorted(row.keys() - self._columns.keys())
            missing = sorted(self._columns.keys() - row.keys())
            raise RecorderError(
                f"signals changed mid-run (added: {added or 'none'}, missing: {missing or 'none'})"
            )
        elif self._columns["t"] and row["t"] <= self._columns["t"][-1]:
            raise RecorderError(f"t must increase: {row['t']} after {self._columns['t'][-1]}")
        for key, value in row.items():
            self._columns[key].append(value)

    def log_contacts(
        self, *, t: float, contacts: Iterable[Contact | tuple[str, str, float, bool]]
    ) -> None:
        self._check_open()
        for c in contacts:
            contact = Contact(*c)
            self._contacts.append((float(t), contact))

    def finish(
        self,
        *,
        success: bool,
        metrics: Mapping[str, float] | None = None,
        failure_reason: str | None = None,
    ) -> Run:
        self._check_open()
        self.finished = True
        return self._recorder._finish_run(self, success, metrics, failure_reason)

    def _check_open(self) -> None:
        if self.finished:
            raise RecorderError(f"run {self.run_id} is already finished")

    def timeseries_table(self) -> pa.Table | None:
        if self._columns is None:
            return None
        return pa.table({k: pa.array(v, pa.float64()) for k, v in self._columns.items()})

    def contacts_table(self) -> pa.Table | None:
        if not self._contacts:
            return None
        return pa.table(
            {
                "t": [t for t, _ in self._contacts],
                "geom_a": [c.geom_a for _, c in self._contacts],
                "geom_b": [c.geom_b for _, c in self._contacts],
                "force_norm": [float(c.force_norm) for _, c in self._contacts],
                "intended": [bool(c.intended) for _, c in self._contacts],
            },
            schema=CONTACT_COLUMNS,
        )


class Recorder:
    """Records runs of one policy version on one suite into a store.

    `suite` is either a path to the suite TOML (registered in the store) or a
    `name@version` already registered there.
    """

    def __init__(
        self,
        store: Store | str | Path,
        policy_version: str,
        suite: str | Path,
        *,
        simulator: SimulatorInfo = UNKNOWN_SIMULATOR,
        physics_hash: str = "unknown",
        git_sha: str | None = None,
    ) -> None:
        if not policy_version:
            raise RecorderError("policy_version must not be empty")
        self.store = store if isinstance(store, Store) else Store(store)
        self.suite = self.store.resolve_suite(suite)
        self.policy_version = policy_version
        self.simulator = simulator
        self.physics_hash = physics_hash
        self.git_sha = git_sha
        self.batch_id: UUID = uuid4()
        self.runs: list[Run] = []
        self._scenario_ids = {scenario_id(sc) for sc in self.suite.scenarios}
        self._closed = False

    @contextmanager
    def run(
        self, *, scenario: Scenario, env_seed: int, policy_seed: int | None = None
    ) -> Iterator[RunContext]:
        """Record one run. If the body raises, the run is stored as failed and the
        exception is suppressed, so one crashing episode does not abort a batch."""
        self._check_open()
        sid = scenario_id(scenario)
        if sid not in self._scenario_ids:
            raise RecorderError(
                f"scenario in family {scenario.family!r} is not part of suite {self.suite.ref}"
            )
        ctx = RunContext(self, sid, scenario.family, env_seed, policy_seed)
        try:
            yield ctx
        except RecorderError:
            raise
        except Exception as exc:
            if ctx.finished:
                raise
            reason = f"exception: {type(exc).__name__}: {exc}"
            log.warning(
                "run in family %s (env_seed %d) raised; recorded as failed: %s",
                scenario.family,
                env_seed,
                reason,
            )
            ctx.finish(success=False, failure_reason=reason)
            return
        if not ctx.finished:
            raise RecorderError("run.finish() was not called before the run block ended")

    def _finish_run(
        self,
        ctx: RunContext,
        success: bool,
        metrics: Mapping[str, float] | None,
        failure_reason: str | None,
    ) -> Run:
        ts = ctx.timeseries_table()
        run = Run(
            run_id=ctx.run_id,
            batch_id=self.batch_id,
            policy_version=self.policy_version,
            suite_ref=self.suite.ref,
            scenario_id=ctx.scenario_id,
            family=ctx.family,
            env_seed=ctx.env_seed,
            policy_seed=ctx.policy_seed,
            simulator=self.simulator,
            physics_hash=self.physics_hash,
            success=success,
            failure_reason=failure_reason,
            metrics=dict(metrics or {}),
            source="native",
            has_timeseries=ts is not None,
            started_at=ctx.started_at,
        )
        if ts is not None:
            self.store.write_timeseries(self.batch_id, run.run_id, ts)
        contacts = ctx.contacts_table()
        if contacts is not None:
            self.store.write_contacts(self.batch_id, run.run_id, contacts)
        self.runs.append(run)
        return run

    def close(self) -> UUID:
        """Write the batch row and return the batch id."""
        self._check_open()
        self._closed = True
        batch = Batch(
            batch_id=self.batch_id,
            policy_version=self.policy_version,
            suite_ref=self.suite.ref,
            created_at=datetime.now(UTC),
            git_sha=self.git_sha,
            n_runs=len(self.runs),
        )
        self.store.write_batch(BatchData(batch=batch, runs=self.runs))
        return self.batch_id

    def _check_open(self) -> None:
        if self._closed:
            raise RecorderError("recorder is closed")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None and not self._closed:
            self.close()
