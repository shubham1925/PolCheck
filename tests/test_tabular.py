from __future__ import annotations

from pathlib import Path

import pytest

from polcheck.config import Config, ReadersConfig, TabularMapping
from polcheck.readers import ReaderError, get_reader
from polcheck.readers.tabular import TabularReader
from polcheck.schema import BatchData, SimulatorInfo
from polcheck.suite import family_scenarios, load_suite
from tests.paths import PICK_SUITE, SHELF_SUITE, TABULAR

PICK_MAPPING = TabularMapping(
    family="family", env_seed="seed", success="success", failure_reason="failure"
)


def config_with(mapping: TabularMapping) -> Config:
    return Config(readers=ReadersConfig(tabular=mapping))


def read(
    path: Path, mapping: TabularMapping = PICK_MAPPING, suite_path: Path = PICK_SUITE
) -> BatchData:
    return TabularReader().read(
        path, policy_version="pick-v3", suite=load_suite(suite_path), config=config_with(mapping)
    )


def summary(data: BatchData) -> list[tuple[object, ...]]:
    return [
        (r.family, r.scenario_id, r.env_seed, r.success, r.failure_reason, r.metrics)
        for r in data.runs
    ]


@pytest.mark.parametrize("name", ["pick.csv", "pick.jsonl"])
def test_reads_fixture(name: str) -> None:
    data = read(TABULAR / name)
    ids = family_scenarios(load_suite(PICK_SUITE))
    assert data.batch.n_runs == 8
    assert data.batch.policy_version == "pick-v3"
    assert data.batch.suite_ref == "pick@1"
    first, second, _, missing_score, fifth, *_ = data.runs
    assert (first.family, first.scenario_id, first.env_seed) == (
        "near-left",
        ids["near-left"][0],
        0,
    )
    assert first.success
    assert first.failure_reason is None
    assert not second.success
    assert second.failure_reason == "dropped"
    assert first.metrics == {"task_score": 0.91, "episode_length": 120.0}  # "notes" is text
    assert missing_score.metrics == {"episode_length": 140.0}
    assert fifth.success  # CSV "1"
    for run in data.runs:
        assert run.source == "tabular"
        assert not run.has_timeseries
        assert run.policy_seed is None
        assert run.simulator == SimulatorInfo(name="unknown", version="unknown")
        assert run.physics_hash == "unknown"


def test_csv_and_jsonl_agree() -> None:
    assert summary(read(TABULAR / "pick.csv")) == summary(read(TABULAR / "pick.jsonl"))


def test_unknown_family_lists_all_of_them() -> None:
    mapping = TabularMapping(env_seed="seed")
    with pytest.raises(ReaderError, match="families not in suite pick@1: far-up, middle"):
        read(TABULAR / "unknown_family.csv", mapping)


def test_explicit_metric_list_and_simulator() -> None:
    mapping = PICK_MAPPING.model_copy(
        update={
            "metrics": ["task_score"],
            "simulator": SimulatorInfo(name="isaac-sim", version="5.0"),
            "physics_hash": "p1",
        }
    )
    data = read(TABULAR / "pick.csv", mapping)
    assert all(set(r.metrics) <= {"task_score"} for r in data.runs)
    assert data.runs[0].simulator.name == "isaac-sim"
    assert data.runs[0].physics_hash == "p1"


def test_missing_columns_reported() -> None:
    mapping = PICK_MAPPING.model_copy(update={"env_seed": "episode_seed", "metrics": ["reward"]})
    with pytest.raises(ReaderError, match="missing columns episode_seed, reward"):
        read(TABULAR / "pick.csv", mapping)


def test_scenario_index_selects_scenario() -> None:
    mapping = TabularMapping(env_seed="seed", scenario_index="scenario")
    data = read(TABULAR / "shelf.csv", mapping, SHELF_SUITE)
    ids = family_scenarios(load_suite(SHELF_SUITE))
    assert [r.scenario_id for r in data.runs] == [ids["shelf"][0], ids["shelf"][1], ids["floor"][0]]
    assert all(r.metrics == {} for r in data.runs)  # the index column is not a metric


def test_scenario_index_required_for_multi_scenario_family() -> None:
    with pytest.raises(ReaderError, match="set scenario_index"):
        read(TABULAR / "shelf.csv", TabularMapping(env_seed="seed"), SHELF_SUITE)


def test_scenario_index_out_of_range(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("family,seed,success,scenario\nshelf,0,true,2\n")
    mapping = TabularMapping(env_seed="seed", scenario_index="scenario")
    with pytest.raises(ReaderError, match="row 1: scenario index 2 out of range"):
        read(path, mapping, SHELF_SUITE)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("near-left,0,maybe", "expected true/false"),
        ("near-left,1.5,true", "expected an integer"),
        (",0,true", "family is empty"),
    ],
)
def test_bad_cells(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(f"family,seed,success\n{body}\n")
    with pytest.raises(ReaderError, match=message):
        read(path, TabularMapping(env_seed="seed"))


def test_nan_metric_omitted_and_inf_rejected(tmp_path: Path) -> None:
    good = tmp_path / "nan.jsonl"
    good.write_text('{"family": "near-left", "seed": 0, "success": true, "score": NaN}\n')
    assert read(good, TabularMapping(env_seed="seed")).runs[0].metrics == {}
    bad = tmp_path / "inf.csv"
    bad.write_text("family,seed,success,score\nnear-left,0,true,inf\n")
    with pytest.raises(ReaderError, match="infinite"):
        read(bad, TabularMapping(env_seed="seed"))


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("runs.parquet", "", "unsupported file type"),
        ("empty.csv", "family,seed,success\n", "no rows"),
        ("broken.jsonl", "{not json\n", "could not parse"),
    ],
)
def test_bad_files(tmp_path: Path, name: str, content: str, message: str) -> None:
    path = tmp_path / name
    path.write_text(content)
    with pytest.raises(ReaderError, match=message):
        read(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ReaderError, match="not found"):
        read(tmp_path / "missing.csv")


def test_registered_as_entry_point() -> None:
    assert isinstance(get_reader("tabular"), TabularReader)
    with pytest.raises(ReaderError, match="available: native, tabular"):
        get_reader("nope")
