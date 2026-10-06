"""M1 demo acceptance: 100 recorded MuJoCo runs and repeatability.

Skipped unless the `[demo]` extra is installed (`uv sync --extra demo`).
"""

from __future__ import annotations

from uuid import UUID

import pytest

pytest.importorskip("gymnasium_robotics")

from demo.run_suite import DEFAULT_SUITE, run_suite
from demo.scripted import ScriptedPick
from polcheck.comparability import check
from polcheck.config import Config
from polcheck.schema import BatchData
from polcheck.store import Store

pytestmark = pytest.mark.demo


def _signals(store: Store, batch_id: UUID) -> list[tuple[object, ...]]:
    """Everything recorded per run except ids and wall-clock start times."""
    out: list[tuple[object, ...]] = []
    for run in store.read_batch(batch_id).runs:
        ts = store.read_timeseries(batch_id, run.run_id)
        contacts = store.read_contacts(batch_id, run.run_id)
        out.append(
            (
                run.family,
                run.env_seed,
                run.success,
                run.failure_reason,
                run.metrics,
                ts.to_pydict() if ts is not None else None,
                contacts.to_pydict() if contacts is not None else None,
            )
        )
    return out


def test_records_100_runs_and_reads_them_back(store: Store) -> None:
    batch_id = run_suite(ScriptedPick(), "scripted-v1", store=store, seeds_per_family=25)
    data = store.read_batch(batch_id)
    assert data.batch.n_runs == 100
    assert {r.family for r in data.runs} == {"near-left", "near-right", "far-left", "far-right"}
    assert len({(r.family, r.env_seed) for r in data.runs}) == 100
    assert len({r.physics_hash for r in data.runs}) == 1
    # The scripted controller is the fallback demo policy; it must be reliable.
    assert sum(r.success for r in data.runs) >= 85

    # Read back through a fresh Store with no index: values match what was recorded.
    fresh = Store(store.root)
    fresh.index_path.unlink()
    assert fresh.read_batch(batch_id) == data
    for run in data.runs:
        assert run.has_timeseries
        ts = fresh.read_timeseries(batch_id, run.run_id)
        assert ts is not None
        names = set(ts.column_names)
        assert {"t", "ee_pos.x", "ee_quat.w", "obj.box.pos.z", "clearance", "qpos.0"} <= names
        assert "joint_torque.0" not in names  # Fetch's arm is mocap-driven; see DECISIONS.md
        assert min(ts.column("clearance").to_pylist()) >= 0.0
        if run.success:
            contacts = fresh.read_contacts(batch_id, run.run_id)
            assert contacts is not None
            assert any(contacts.column("intended").to_pylist())


def test_same_seed_same_policy_is_repeatable(store: Store) -> None:
    families = ["near-left", "far-right"]
    first = run_suite(ScriptedPick(), "a", store=store, seeds_per_family=5, families=families)
    second = run_suite(ScriptedPick(), "b", store=store, seeds_per_family=5, families=families)
    assert _signals(store, first) == _signals(store, second)


def test_suite_file_is_the_documented_one() -> None:
    assert DEFAULT_SUITE.name == "pick.toml"


def test_two_recorded_batches_are_comparable_and_paired(store: Store) -> None:
    def record(version: str) -> BatchData:
        batch_id = run_suite(
            ScriptedPick(), version, store=store, seeds_per_family=3, repeat_check=2
        )
        return store.read_batch(batch_id)

    base, cand = record("v1"), record("v1-again")
    assert base.batch.deterministic is True
    assert base.batch.n_runs == 4 * 3 + 2
    result = check(base, cand, Config(min_n=3))
    assert result.status == "comparable"
    assert result.paired
    assert result.issues == []
    assert [(f.n_base, f.n_cand) for f in result.families] == [(3, 3)] * 4  # repeats counted once
