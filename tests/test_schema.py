from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel, JsonValue, ValidationError

from polcheck.schema import (
    Batch,
    BatchManifest,
    ManifestContents,
    Run,
    Scenario,
    SeedRange,
    SimulatorInfo,
    Suite,
)

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
BATCH_ID = uuid4()


def make_run(**overrides: Any) -> Run:
    fields: dict[str, Any] = {
        "run_id": uuid4(),
        "batch_id": BATCH_ID,
        "policy_version": "pick-v14",
        "suite_ref": "pick@1",
        "scenario_id": "0123456789abcdef",
        "family": "near-left",
        "env_seed": 4211,
        "policy_seed": 7,
        "simulator": SimulatorInfo(name="mujoco", version="3.1.6"),
        "physics_hash": "a1b2c3",
        "success": True,
        "metrics": {"task_time": 6.2},
        "source": "native",
        "has_timeseries": True,
        "started_at": NOW,
    }
    fields.update(overrides)
    return Run(**fields)


def make_batch(**overrides: Any) -> Batch:
    fields: dict[str, Any] = {
        "batch_id": BATCH_ID,
        "policy_version": "pick-v14",
        "suite_ref": "pick@1",
        "created_at": NOW,
        "n_runs": 200,
    }
    fields.update(overrides)
    return Batch(**fields)


def make_suite() -> Suite:
    return Suite(
        name="pick",
        version=1,
        scenarios=[
            Scenario(
                family="near-left",
                config={"object_x": [0.0, 0.15], "object_y": [0.0, 0.15], "table_height": 0.4},
                seeds=SeedRange(start=0, count=50),
            )
        ],
    )


def assert_round_trips(model: BaseModel) -> None:
    restored = type(model).model_validate_json(model.model_dump_json())
    assert restored == model


@pytest.mark.parametrize(
    "model",
    [
        make_suite(),
        make_run(),
        make_run(policy_seed=None, success=False, failure_reason="exception: boom", metrics={}),
        make_run(source="tabular", has_timeseries=False),
        make_batch(),
        make_batch(git_sha="deadbeef", deterministic=True),
        BatchManifest(
            batch=make_batch(),
            suite_toml='[suite]\nname = "pick"\nversion = 1\n',
            measure_versions={"sparc": "1"},
            contents=ManifestContents(timeseries=True, contacts=True),
            is_baseline=True,
        ),
    ],
    ids=lambda m: type(m).__name__,
)
def test_models_round_trip_to_json(model: BaseModel) -> None:
    assert_round_trips(model)


json_values: st.SearchStrategy[JsonValue] = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda children: st.lists(children) | st.dictionaries(st.text(), children),
    max_leaves=20,
)


@given(config=st.dictionaries(st.text(), json_values, max_size=5))
def test_scenario_config_round_trips(config: dict[str, JsonValue]) -> None:
    assert_round_trips(Scenario(family="f", config=config, seeds=SeedRange(start=0, count=1)))


@given(
    metrics=st.dictionaries(
        st.text(min_size=1), st.floats(allow_nan=False, allow_infinity=False), max_size=10
    )
)
def test_run_metrics_round_trip(metrics: dict[str, float]) -> None:
    assert_round_trips(make_run(metrics=metrics))


def test_suite_ref_property() -> None:
    assert make_suite().ref == "pick@1"


def test_seed_range() -> None:
    assert list(SeedRange(start=3, count=2).seeds()) == [3, 4]


@pytest.mark.parametrize("bad", ["pick", "pick@0", "pick@v1", "@1", "pi@ck@1"])
def test_bad_suite_ref_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        make_run(suite_ref=bad)


@pytest.mark.parametrize("bad", ["0123456789ABCDEF", "0123", "0123456789abcdefg"])
def test_bad_scenario_id_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        make_run(scenario_id=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_metric_rejected(bad: float) -> None:
    with pytest.raises(ValidationError):
        make_run(metrics={"sparc": bad})


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        make_run(started_at=datetime(2026, 10, 4, 12, 0))


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        make_run(colour="red")


def test_unknown_source_rejected() -> None:
    with pytest.raises(ValidationError):
        make_run(source="isaac")


def test_models_are_frozen() -> None:
    run = make_run()
    with pytest.raises(ValidationError):
        run.success = False  # type: ignore[misc]
