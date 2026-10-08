from __future__ import annotations

import logging

import numpy as np
import pytest

from polcheck.recorder import Contact, Recorder, RecorderError
from polcheck.schema import Scenario, SeedRange, SimulatorInfo
from polcheck.store import Store
from tests.paths import PICK_SUITE

SIM = SimulatorInfo(name="mujoco", version="3.1.6")


def make_recorder(store: Store, policy_version: str = "pick-v1") -> Recorder:
    return Recorder(store, policy_version, PICK_SUITE, simulator=SIM, physics_hash="ph1")


def test_recorded_values_read_back_identically(store: Store) -> None:
    rng = np.random.default_rng(0)
    rec = make_recorder(store)
    expected: dict[str, dict[str, list[float]]] = {}
    for i, scenario in enumerate(rec.suite.scenarios):
        steps = 5 + i
        signals = {
            "joint_pos": rng.normal(size=(steps, 7)),
            "joint_vel": rng.normal(size=(steps, 7)),
            "ee_pos": rng.normal(size=(steps, 3)),
            "ee_quat": rng.normal(size=(steps, 4)),
            "box_pos": rng.normal(size=(steps, 3)),
            "box_quat": rng.normal(size=(steps, 4)),
            "clearance": rng.uniform(size=steps),
            "qpos": rng.normal(size=(steps, 15)),
        }
        with rec.run(scenario=scenario, env_seed=i, policy_seed=None) as run:
            for k in range(steps):
                run.log(
                    t=k * 0.04,
                    joint_pos=signals["joint_pos"][k],
                    joint_vel=signals["joint_vel"][k],
                    ee_pos=signals["ee_pos"][k],
                    ee_quat=signals["ee_quat"][k],
                    objects={"box": (signals["box_pos"][k], signals["box_quat"][k])},
                    clearance=float(signals["clearance"][k]),
                    qpos=signals["qpos"][k],
                )
            run.log_contacts(t=0.04, contacts=[Contact("finger_l", "box", 2.5, True)])
            run.log_contacts(t=0.08, contacts=[("arm", "table", 7.25, False)])
            recorded = run.finish(success=i != 2, metrics={"goal_distance": 0.01 * i})
        expected[str(recorded.run_id)] = {
            "t": [k * 0.04 for k in range(steps)],
            "joint_pos.6": signals["joint_pos"][:, 6].tolist(),
            "ee_quat.w": signals["ee_quat"][:, 0].tolist(),
            "obj.box.pos.z": signals["box_pos"][:, 2].tolist(),
            "obj.box.quat.x": signals["box_quat"][:, 1].tolist(),
            "clearance": signals["clearance"].tolist(),
            "qpos.14": signals["qpos"][:, 14].tolist(),
        }
    batch_id = rec.close()

    data = store.read_batch(batch_id)
    assert data.runs == rec.runs
    assert data.batch.n_runs == 4
    for stored in data.runs:
        ts = store.read_timeseries(batch_id, stored.run_id)
        assert ts is not None
        assert ts.num_columns == 1 + 7 + 7 + 3 + 4 + 3 + 4 + 1 + 15
        for column, values in expected[str(stored.run_id)].items():
            assert ts.column(column).to_pylist() == values  # exact, not approximate
        contacts = store.read_contacts(batch_id, stored.run_id)
        assert contacts is not None
        assert contacts.to_pylist() == [
            {"t": 0.04, "geom_a": "finger_l", "geom_b": "box", "force_norm": 2.5, "intended": True},
            {"t": 0.08, "geom_a": "arm", "geom_b": "table", "force_norm": 7.25, "intended": False},
        ]
    assert {r.success for r in data.runs} == {True, False}
    assert all(r.has_timeseries and r.source == "native" for r in data.runs)
    assert all(r.simulator == SIM and r.physics_hash == "ph1" for r in data.runs)
    assert all(r.sample_rate_hz == pytest.approx(25.0) for r in data.runs)


def test_exception_records_failed_run(store: Store, caplog: pytest.LogCaptureFixture) -> None:
    rec = make_recorder(store)
    scenario = rec.suite.scenarios[0]
    with caplog.at_level(logging.WARNING), rec.run(scenario=scenario, env_seed=3) as run:
        run.log(t=0.0, ee_pos=[0, 0, 0])
        raise ValueError("simulator exploded")
    with rec.run(scenario=scenario, env_seed=4) as run:  # the batch carries on
        run.finish(success=True)
    batch_id = rec.close()

    failed, ok = store.read_batch(batch_id).runs
    assert not failed.success
    assert failed.failure_reason == "exception: ValueError: simulator exploded"
    assert failed.has_timeseries  # partial time-series is kept
    assert ok.success
    assert "recorded as failed" in caplog.text


def test_recorder_misuse_is_not_swallowed(store: Store) -> None:
    rec = make_recorder(store)
    scenario = rec.suite.scenarios[0]
    with (
        pytest.raises(RecorderError, match="signals changed"),
        rec.run(scenario=scenario, env_seed=0) as run,
    ):
        run.log(t=0.0, ee_pos=[0, 0, 0])
        run.log(t=0.1, ee_pos=[0, 0, 0], clearance=0.2)
    assert rec.runs == []


def test_invalid_metrics_raise_instead_of_recording_failure(store: Store) -> None:
    rec = make_recorder(store)
    with (
        pytest.raises(ValueError, match="finite"),
        rec.run(scenario=rec.suite.scenarios[0], env_seed=0) as run,
    ):
        run.finish(success=True, metrics={"score": float("nan")})
    assert rec.runs == []


def test_finish_is_required(store: Store) -> None:
    rec = make_recorder(store)
    with (
        pytest.raises(RecorderError, match="finish"),
        rec.run(scenario=rec.suite.scenarios[0], env_seed=0) as run,
    ):
        run.log(t=0.0, ee_pos=[0, 0, 0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ee_pos": [0, 0]}, "3 elements"),
        ({"ee_quat": [1, 0, 0]}, "4 elements"),
        ({"joint_pos": [[0, 1], [2, 3]]}, "1-D"),
        ({"objects": {"box": ([0, 0, 0], [1, 0, 0])}}, "quaternion"),
    ],
)
def test_bad_signal_shapes(store: Store, kwargs: dict[str, object], message: str) -> None:
    rec = make_recorder(store)
    with (
        pytest.raises(RecorderError, match=message),
        rec.run(scenario=rec.suite.scenarios[0], env_seed=0) as run,
    ):
        run.log(t=0.0, **kwargs)  # type: ignore[arg-type]


def test_time_must_increase(store: Store) -> None:
    rec = make_recorder(store)
    with (
        pytest.raises(RecorderError, match="t must increase"),
        rec.run(scenario=rec.suite.scenarios[0], env_seed=0) as run,
    ):
        run.log(t=0.1, clearance=1.0)
        run.log(t=0.1, clearance=1.0)


def test_scenario_must_belong_to_suite(store: Store) -> None:
    rec = make_recorder(store)
    stranger = Scenario(family="near-left", config={"other": 1}, seeds=SeedRange(start=0, count=1))
    with (
        pytest.raises(RecorderError, match="not part of suite"),
        rec.run(scenario=stranger, env_seed=0),
    ):
        pass


def test_run_without_signals(store: Store) -> None:
    rec = make_recorder(store)
    with rec.run(scenario=rec.suite.scenarios[0], env_seed=0) as run:
        run.finish(success=True, metrics={"task_time": 6.2})
    batch_id = rec.close()
    (run_,) = store.read_batch(batch_id).runs
    assert not run_.has_timeseries
    assert run_.sample_rate_hz is None
    assert store.read_timeseries(batch_id, run_.run_id) is None


def test_sample_rate_is_the_median_step(store: Store) -> None:
    rec = make_recorder(store)
    scenario = rec.suite.scenarios[0]
    with rec.run(scenario=scenario, env_seed=0) as run:
        for t in (0.0, 0.02, 0.04, 0.10, 0.12):  # one dropped-frame gap
            run.log(t=t, ee_pos=[0.0, 0.0, 0.0])
        steady = run.finish(success=True)
    with rec.run(scenario=scenario, env_seed=1) as run:
        run.log(t=0.0, ee_pos=[0.0, 0.0, 0.0])
        single = run.finish(success=True)
    assert steady.sample_rate_hz == pytest.approx(50.0)
    assert single.has_timeseries
    assert single.sample_rate_hz is None


def test_closed_recorder(store: Store) -> None:
    rec = make_recorder(store)
    rec.close()
    with pytest.raises(RecorderError, match="closed"):
        rec.close()
    with (
        pytest.raises(RecorderError, match="closed"),
        rec.run(scenario=rec.suite.scenarios[0], env_seed=0),
    ):
        pass


def test_context_manager_closes_and_resolves_registered_suite(store: Store) -> None:
    with make_recorder(store) as rec, rec.run(scenario=rec.suite.scenarios[1], env_seed=9) as run:
        run.finish(success=True)
    # Once registered, the suite can be referenced by name@version.
    second = Recorder(store, "pick-v2", "pick@1")
    assert second.suite == rec.suite
    assert [b.batch_id for b in store.list_batches()] == [rec.batch_id]


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        ([(0, True), (1, True)], None),  # no repeated seed: no evidence
        ([(0, True), (0, True)], True),  # identical repeat
        ([(0, True), (0, False)], False),  # repeat disagrees
    ],
)
def test_close_records_determinism(
    store: Store, outcomes: list[tuple[int, bool]], expected: bool | None
) -> None:
    rec = make_recorder(store)
    for env_seed, success in outcomes:
        with rec.run(scenario=rec.suite.scenarios[0], env_seed=env_seed) as run:
            run.finish(success=success)
    batch_id = rec.close()
    assert store.read_batch_meta(batch_id).deterministic is expected
