"""M1 demo acceptance: 100 recorded MuJoCo runs and repeatability.

Skipped unless the `[demo]` extra is installed (`uv sync --extra demo`).
"""

from __future__ import annotations

from uuid import UUID

import pytest

pytest.importorskip("gymnasium_robotics")

import numpy as np

from demo.envs import Perception
from demo.run_suite import DEFAULT_SUITE, run_suite
from demo.scripted import ScriptedPick
from polcheck.comparability import check
from polcheck.config import Config
from polcheck.schema import BatchData, Scenario, SeedRange
from polcheck.store import Store
from polcheck.suite import load_suite

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


def test_default_suite_is_pick_v2() -> None:
    suite = load_suite(DEFAULT_SUITE)
    assert suite.ref == "pick@2"
    assert {sc.config["perception_noise"] for sc in suite.scenarios} == {0.012}


def _obs() -> dict[str, np.ndarray]:
    return {
        "observation": np.arange(25, dtype=np.float64),
        "achieved_goal": np.zeros(3),
        "desired_goal": np.ones(3),
    }


def test_perception_is_seeded_and_only_moves_the_object() -> None:
    a, b, c = Perception(0.012, 7), Perception(0.012, 7), Perception(0.012, 8)
    seen_a, seen_b, seen_c = a(_obs()), b(_obs()), c(_obs())
    assert np.array_equal(seen_a["observation"], seen_b["observation"])  # same seed, same error
    assert not np.array_equal(seen_a["observation"], seen_c["observation"])
    error = seen_a["observation"] - _obs()["observation"]
    assert np.array_equal(error[3:6], error[6:9])  # position and relative position agree
    assert np.all(error[3:6] != 0)  # the object position is perturbed
    assert np.count_nonzero(np.delete(error, range(3, 9))) == 0  # nothing else touched
    assert np.array_equal(seen_a["achieved_goal"], np.zeros(3))  # success uses the truth


def test_zero_noise_passes_observation_through() -> None:
    obs = _obs()
    assert Perception(0.0, 1)(obs) is obs


@pytest.mark.parametrize("bad", [-0.01, "loud", True])
def test_invalid_perception_noise(bad: object) -> None:
    scenario = Scenario(
        family="f",
        config={"object_x": [0, 0], "object_y": [0, 0], "perception_noise": bad},  # type: ignore[dict-item]
        seeds=SeedRange(start=0, count=1),
    )
    with pytest.raises(ValueError, match="perception_noise"):
        Perception.for_scenario(scenario, 0)


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
