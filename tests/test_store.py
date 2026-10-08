from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from polcheck.schema import Batch, BatchData, Run, SimulatorInfo
from polcheck.store import Store, StoreError, SuiteFrozenError
from polcheck.suite import load_suite, scenario_id
from tests.paths import PICK_SUITE

T0 = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def make_batch_data(
    policy_version: str = "pick-v1",
    *,
    created_at: datetime = T0,
    n: int = 4,
) -> BatchData:
    suite = load_suite(PICK_SUITE)
    batch_id = uuid4()
    runs = [
        Run(
            run_id=uuid4(),
            batch_id=batch_id,
            policy_version=policy_version,
            suite_ref=suite.ref,
            scenario_id=scenario_id(suite.scenarios[i % 4]),
            family=suite.scenarios[i % 4].family,
            env_seed=i,
            policy_seed=None if i % 2 else 7,
            simulator=SimulatorInfo(name="mujoco", version="3.1.6"),
            physics_hash="abc",
            success=i % 3 != 0,
            failure_reason="timeout" if i % 3 == 0 else None,
            # metrics differ between runs, and one run has none
            metrics={} if i == 1 else {"task_time": 5.0 + i / 7, "score": i * 0.1},
            source="native",
            has_timeseries=i != 0,
            sample_rate_hz=None if i == 0 else 25.0 * i,
            # a non-UTC timezone must come back as the same instant
            started_at=(T0 + timedelta(seconds=i)).astimezone(timezone(timedelta(hours=2))),
        )
        for i in range(n)
    ]
    batch = Batch(
        batch_id=batch_id,
        policy_version=policy_version,
        suite_ref=suite.ref,
        created_at=created_at,
        git_sha="deadbeef",
        n_runs=n,
    )
    return BatchData(batch=batch, runs=runs)


def test_batch_round_trip(store: Store) -> None:
    store.register_suite(PICK_SUITE)
    data = make_batch_data()
    store.write_batch(data)
    assert store.read_batch(data.batch.batch_id) == data
    assert store.read_batch_meta(data.batch.batch_id) == data.batch


def test_empty_batch_round_trip(store: Store) -> None:
    store.register_suite(PICK_SUITE)
    data = make_batch_data(n=0)
    store.write_batch(data)
    assert store.read_batch(data.batch.batch_id) == data


def test_write_requires_registered_suite(store: Store) -> None:
    with pytest.raises(StoreError, match="not registered"):
        store.write_batch(make_batch_data())


def test_duplicate_batch_rejected(store: Store) -> None:
    store.register_suite(PICK_SUITE)
    data = make_batch_data()
    store.write_batch(data)
    with pytest.raises(StoreError, match="already exists"):
        store.write_batch(data)


def test_timeseries_and_contacts(store: Store) -> None:
    batch_id, run_id = uuid4(), uuid4()
    ts = pa.table({"t": [0.0, 0.1], "ee_pos.x": [1.0, 1.5]})
    contacts = pa.table(
        {
            "t": [0.1],
            "geom_a": ["finger"],
            "geom_b": ["box"],
            "force_norm": [3.5],
            "intended": [True],
        }
    )
    store.write_timeseries(batch_id, run_id, ts)
    store.write_contacts(batch_id, run_id, contacts)
    assert store.read_timeseries(batch_id, run_id) == ts
    assert store.read_contacts(batch_id, run_id) == contacts
    assert store.read_timeseries(batch_id, uuid4()) is None


def _query_snapshot(store: Store) -> tuple[object, ...]:
    return (
        store.list_batches(),
        store.query(
            'SELECT run_id, family, env_seed, success, "metric.task_time", "metric.score" '
            "FROM runs ORDER BY run_id"
        ),
        store.query("SELECT family, count(*), avg(success::INT) FROM runs GROUP BY 1 ORDER BY 1"),
        store.query("SELECT * FROM baselines"),
    )


def test_reindex_restores_all_queries(store: Store) -> None:
    store.register_suite(PICK_SUITE)
    first = make_batch_data("pick-v1", created_at=T0)
    second = make_batch_data("pick-v2", created_at=T0 + timedelta(hours=1))
    store.write_batch(first)
    store.write_batch(second)
    store.set_baseline(first.batch.batch_id)
    before = _query_snapshot(store)

    store.index_path.unlink()
    assert store.reindex() == 2
    assert _query_snapshot(store) == before

    # A missing index is also rebuilt on first use.
    store.index_path.unlink()
    assert _query_snapshot(store) == before


def test_batch_without_sample_rate_column_still_reads(store: Store) -> None:
    """Batches written before runs stored `sample_rate_hz` read back as None."""
    store.register_suite(PICK_SUITE)
    data = make_batch_data()
    store.write_batch(data)
    path = store.batch_dir(data.batch.batch_id) / "runs.parquet"
    table = pq.read_table(path)
    pq.write_table(table.drop_columns(["sample_rate_hz"]), path)

    assert all(r.sample_rate_hz is None for r in store.read_batch(data.batch.batch_id).runs)
    assert store.reindex() == 1
    assert store.query("SELECT count(*) FROM runs WHERE sample_rate_hz IS NULL") == [(4,)]


def test_reindex_ignores_incomplete_batches(store: Store) -> None:
    store.register_suite(PICK_SUITE)
    data = make_batch_data()
    store.write_batch(data)
    partial = store.batch_dir(uuid4())
    (partial / "ts").mkdir(parents=True)  # e.g. recorder crashed before close()
    assert store.reindex() == 1


def test_list_and_latest_batch(store: Store) -> None:
    store.register_suite(PICK_SUITE)
    old = make_batch_data("pick-v1", created_at=T0)
    new = make_batch_data("pick-v1", created_at=T0 + timedelta(days=1))
    other = make_batch_data("pick-v2", created_at=T0 + timedelta(days=2))
    for data in (new, other, old):
        store.write_batch(data)
    assert [b.batch_id for b in store.list_batches()] == [
        old.batch.batch_id,
        new.batch.batch_id,
        other.batch.batch_id,
    ]
    assert store.list_batches(policy_version="pick-v2") == [other.batch]
    latest = store.latest_batch("pick-v1")
    assert latest is not None
    assert latest.batch_id == new.batch.batch_id
    assert store.latest_batch("pick-v9") is None


def test_register_suite_is_idempotent(store: Store, tmp_path: Path) -> None:
    store.register_suite(PICK_SUITE)
    reformatted = tmp_path / "pick.toml"
    reformatted.write_text("# same content\n" + PICK_SUITE.read_text())
    assert store.register_suite(reformatted) == store.get_suite("pick@1")


def _changed_suite(tmp_path: Path) -> Path:
    changed = tmp_path / "pick_changed.toml"
    changed.write_text(PICK_SUITE.read_text().replace("count = 4", "count = 5", 1))
    return changed


def test_unfrozen_suite_can_change(store: Store, tmp_path: Path) -> None:
    store.register_suite(PICK_SUITE)
    store.register_suite(_changed_suite(tmp_path))
    assert store.get_suite("pick@1").scenarios[0].seeds.count == 5


def test_baseline_freezes_suite(store: Store, tmp_path: Path) -> None:
    store.register_suite(PICK_SUITE)
    data = make_batch_data()
    store.write_batch(data)
    store.set_baseline(data.batch.batch_id)
    assert store.baseline("pick@1") == data.batch.batch_id
    assert store.is_frozen("pick@1")
    with pytest.raises(SuiteFrozenError, match="Bump the version"):
        store.register_suite(_changed_suite(tmp_path))
    store.register_suite(PICK_SUITE)  # unchanged content is still fine


def test_baselines_survive_reindex(store: Store) -> None:
    store.register_suite(PICK_SUITE)
    data = make_batch_data()
    store.write_batch(data)
    store.set_baseline(data.batch.batch_id)
    store.index_path.unlink()
    store.reindex()
    assert store.query("SELECT suite_ref, batch_id FROM baselines") == [
        ("pick@1", str(data.batch.batch_id))
    ]


def test_resolve_suite(store: Store, tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="not registered"):
        store.resolve_suite("pick@1")
    assert store.resolve_suite(str(PICK_SUITE)).ref == "pick@1"
    assert store.resolve_suite("pick@1").ref == "pick@1"
    with pytest.raises(StoreError, match="not found"):
        store.resolve_suite(tmp_path / "missing.toml")


def test_store_is_relocatable(store: Store, tmp_path: Path) -> None:
    store.register_suite(PICK_SUITE)
    data = make_batch_data()
    store.write_batch(data)
    moved = Store(tmp_path / "moved")
    shutil.copytree(store.root, moved.root)
    (moved.root / "index.duckdb").unlink()
    assert moved.read_batch(data.batch.batch_id) == data
    assert [b.batch_id for b in moved.list_batches()] == [data.batch.batch_id]


def test_batch_ids_are_uuids(store: Store) -> None:
    store.register_suite(PICK_SUITE)
    data = make_batch_data()
    store.write_batch(data)
    (row,) = store.query("SELECT batch_id FROM batches")
    assert UUID(row[0]) == data.batch.batch_id
