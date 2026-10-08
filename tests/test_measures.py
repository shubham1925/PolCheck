from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pytest

from polcheck import Contact, Recorder, RunData, measure
from polcheck.cli import EXIT_USAGE, main
from polcheck.config import Config, MeasureConfig
from polcheck.measures import MeasureError, TimeSeries, load_registry
from polcheck.measures.compute import FAILED_RUN, MeasureValues, measure_batch
from polcheck.recorder import RecorderError
from polcheck.schema import BatchData
from polcheck.store import Store
from tests.paths import PICK_SUITE

DT = 0.04

# --- helpers ------------------------------------------------------------------


def min_jerk(t: np.ndarray, distance: float = 0.3) -> np.ndarray:
    """A smooth point-to-point reach along x."""
    s = (t - t[0]) / (t[-1] - t[0])
    x = distance * (10 * s**3 - 15 * s**4 + 6 * s**5)
    return np.column_stack([x, np.zeros_like(t), np.zeros_like(t)])


def jerky(t: np.ndarray) -> np.ndarray:
    """The same reach with a 6 Hz, 3 mm wobble along the path."""
    pos = min_jerk(t)
    pos[:, 0] += 0.003 * np.sin(2 * np.pi * 6 * t)
    return pos


Episode = dict[str, Any]


def episode(
    pos: np.ndarray,
    *,
    success: bool = True,
    clearance: np.ndarray | None = None,
    contacts: list[Contact] | None = None,
    track_contacts: bool = True,
    signals: dict[str, np.ndarray] | None = None,
    metrics: dict[str, float] | None = None,
) -> Episode:
    return {
        "pos": pos,
        "success": success,
        "clearance": clearance,
        "contacts": contacts or [],
        "track_contacts": track_contacts,
        "signals": signals,
        "metrics": metrics,
    }


def record(store: Store, episodes: list[Episode], version: str = "v1") -> BatchData:
    rec = Recorder(store, version, PICK_SUITE)
    scenario = rec.suite.scenarios[0]
    for seed, ep in enumerate(episodes):
        with rec.run(scenario=scenario, env_seed=seed) as run:
            for k, p in enumerate(ep["pos"]):
                kwargs: dict[str, Any] = {"t": k * DT, "ee_pos": p}
                if ep["clearance"] is not None:
                    kwargs["clearance"] = float(ep["clearance"][k])
                if ep["signals"] is not None:
                    kwargs["signals"] = {name: v[k] for name, v in ep["signals"].items()}
                run.log(**kwargs)
                if ep["track_contacts"]:
                    run.log_contacts(t=k * DT, contacts=ep["contacts"] if k == 1 else [])
            run.finish(success=ep["success"], metrics=ep["metrics"])
    return store.read_batch(rec.close())


def values(results: dict[str, MeasureValues], name: str) -> list[float | None]:
    return list(results[name].values.values())


def compute(
    store: Store, data: BatchData, config: Config | None = None
) -> dict[str, MeasureValues]:
    return measure_batch(store, data, store.get_suite("pick@1"), load_registry(config or Config()))


def defined(results: dict[str, MeasureValues], name: str) -> list[float]:
    out = values(results, name)
    assert None not in out, results[name].skipped
    return [v for v in out if v is not None]


def threshold(results: dict[str, MeasureValues], name: str) -> float:
    value = results[name].spec.threshold
    assert value is not None
    return value


T = np.arange(0, 2.0, DT)

# --- built-in measures --------------------------------------------------------


def test_jerky_trajectory_scores_worse_on_both_smoothness_measures(store: Store) -> None:
    data = record(store, [episode(min_jerk(T)), episode(jerky(T))])
    results = compute(store, data)
    smooth_sparc, jerky_sparc = defined(results, "sparc")
    smooth_ldlj, jerky_ldlj = defined(results, "log_dimensionless_jerk")
    sparc_threshold = threshold(results, "sparc")
    ldlj_threshold = threshold(results, "log_dimensionless_jerk")
    # "Clearly worse": by more than each measure's own threshold.
    assert jerky_sparc < smooth_sparc - sparc_threshold
    assert jerky_ldlj > smooth_ldlj + ldlj_threshold


def test_hesitation_time_counts_a_pause_inside_the_trimmed_window(store: Store) -> None:
    move = np.arange(0, 1.0, DT) * 0.2  # 0.2 m/s for 1 s
    path = np.concatenate([move, np.full(25, move[-1] + 0.2 * DT), move[-1] + 0.2 * DT + move])
    pos = np.column_stack([path, np.zeros_like(path), np.zeros_like(path)])
    data = record(store, [episode(pos)])
    (hesitation,) = values(compute(store, data), "hesitation_time")
    assert hesitation == pytest.approx(1.0, abs=1e-9)  # the 1 s pause; the trimmed ends move


def test_hesitation_ignores_stillness_in_first_and_last_tenth(store: Store) -> None:
    still = np.zeros(5)  # 0.16 s still at the start, inside the first 10% of 2.76 s
    path = np.concatenate([still, np.arange(1, 65) * 0.2 * DT])
    pos = np.column_stack([path, np.zeros_like(path), np.zeros_like(path)])
    (hesitation,) = values(compute(store, record(store, [episode(pos)])), "hesitation_time")
    assert hesitation == 0.0


def test_contact_force_clearance_and_task_time(store: Store) -> None:
    contacts = [
        Contact("finger", "box", 30.0, True),  # intended: ignored however hard
        Contact("wrist", "table", 7.5, False),
        Contact("elbow", "shelf", 2.0, False),
    ]
    clearance = np.linspace(0.2, 0.05, len(T))
    data = record(
        store,
        [
            episode(min_jerk(T), contacts=contacts, clearance=clearance),
            episode(min_jerk(T)),  # contacts tracked, none happened
            episode(min_jerk(T), track_contacts=False, success=False),
        ],
    )
    results = compute(store, data)
    assert values(results, "peak_unintended_contact_force") == [7.5, 0.0, None]
    assert values(results, "min_clearance")[0] == pytest.approx(0.05)
    assert values(results, "task_time")[:2] == [pytest.approx(T[-1])] * 2

    third = data.runs[2].run_id
    assert results["peak_unintended_contact_force"].skipped[third] == "missing signal(s): contacts"
    assert results["min_clearance"].skipped[third] == "missing signal(s): clearance"
    assert results["task_time"].skipped[third] == FAILED_RUN


def test_runs_without_timeseries_are_skipped_not_errors(store: Store) -> None:
    rec = Recorder(store, "v1", PICK_SUITE)
    with rec.run(scenario=rec.suite.scenarios[0], env_seed=0) as run:
        run.finish(success=True)
    data = store.read_batch(rec.close())
    results = compute(store, data)
    for name in (
        "sparc",
        "log_dimensionless_jerk",
        "min_clearance",
        "hesitation_time",
        "task_time",
    ):
        assert values(results, name) == [None]
        assert "missing signal(s)" in results[name].skipped[data.runs[0].run_id]


def test_standing_still_has_no_smoothness_value(store: Store) -> None:
    data = record(store, [episode(np.zeros((len(T), 3)))])
    results = compute(store, data)
    assert values(results, "sparc") == [None]
    assert results["sparc"].skipped[data.runs[0].run_id] == "measure returned no value"


# --- custom measures, signals and the registry ----------------------------------

CUSTOM = '''
from polcheck import RunData, measure


@measure(name="peak_wrist_force", worse="higher", threshold=5.0, unit="N",
         requires=["sig.wrist_force"])
def peak_wrist_force(run: RunData) -> float:
    """Largest wrist force-sensor reading."""
    return float(run.ts["sig.wrist_force"].max())


@measure(name="always_fails", worse="higher", threshold=1.0)
def always_fails(run: RunData) -> float:
    raise RuntimeError("boom")
'''


@pytest.fixture
def custom_file(tmp_path: Path) -> Path:
    path = tmp_path / "my_measures.py"
    path.write_text(CUSTOM)
    return path


def test_custom_signal_and_custom_measure_from_file(
    store: Store, custom_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    force = np.linspace(1.0, 41.2, len(T))
    data = record(store, [episode(min_jerk(T), signals={"wrist_force": force})])
    ts = store.read_timeseries(data.batch.batch_id, data.runs[0].run_id)
    assert ts is not None
    assert "sig.wrist_force" in ts.column_names

    with caplog.at_level(logging.WARNING):
        results = compute(store, data, Config(measure_paths=[custom_file]))
    assert values(results, "peak_wrist_force") == [pytest.approx(41.2)]
    assert results["peak_wrist_force"].spec.source == "file"
    # A measure that raises is skipped with the error, and the others still run.
    assert values(results, "always_fails") == [None]
    assert results["always_fails"].skipped[data.runs[0].run_id] == "error: RuntimeError: boom"
    assert "always_fails raised on 1 run(s)" in caplog.text
    assert values(results, "sparc")[0] is not None


def test_vector_signals_and_timeseries_lookup(store: Store) -> None:
    six_axis = np.tile(np.arange(6.0), (len(T), 1))
    data = record(store, [episode(min_jerk(T), signals={"ft": six_axis})])
    ts = TimeSeries(store.read_timeseries(data.batch.batch_id, data.runs[0].run_id))
    assert ts["sig.ft"].shape == (len(T), 6)
    assert ts["ee_pos"].shape == (len(T), 3)
    assert ts["ee_pos.x"].shape == (len(T),)
    assert ts.has("sig.ft")
    assert not ts.has("sig.f")  # prefixes match whole name segments only
    with pytest.raises(KeyError, match="no signal"):
        ts["joint_pos"]


@pytest.mark.parametrize("bad", ["1force", "wrist.force", "wrist-force", ""])
def test_bad_signal_names_rejected(store: Store, bad: str) -> None:
    rec = Recorder(store, "v1", PICK_SUITE)
    with (
        pytest.raises(RecorderError, match="signal name"),
        rec.run(scenario=rec.suite.scenarios[0], env_seed=0) as run,
    ):
        run.log(t=0.0, signals={bad: 1.0})


def test_entry_point_measures(monkeypatch: pytest.MonkeyPatch, custom_file: Path) -> None:
    module = load_registry(Config(measure_paths=[custom_file]))["peak_wrist_force"].func
    assert module is not None

    class FakeEntryPoint:
        name, value = "acme", "acme_measures:peak_wrist_force"

        def load(self) -> Any:
            return module

    monkeypatch.setattr("polcheck.measures.entry_points", lambda group: [FakeEntryPoint()])
    spec = load_registry(Config())["peak_wrist_force"]
    assert (spec.source, spec.origin) == ("entry_point", "acme_measures:peak_wrist_force")


def test_duplicate_measure_names_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dupe.py"
    path.write_text(
        "from polcheck import measure\n"
        "@measure(name='sparc', worse='lower', threshold=0.1)\n"
        "def my_sparc(run): return 0.0\n"
    )
    with pytest.raises(MeasureError, match="'sparc' is defined twice"):
        load_registry(Config(measure_paths=[path]))


def test_importing_a_builtin_into_a_file_does_not_duplicate_it(tmp_path: Path) -> None:
    path = tmp_path / "reuse.py"
    path.write_text("from polcheck.measures.builtin import sparc_measure  # noqa: F401\n")
    assert "sparc" in load_registry(Config(measure_paths=[path]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "bad name", "worse": "higher", "threshold": 1.0}, "must start with a letter"),
        ({"name": "m", "worse": "up", "threshold": 1.0}, "worse must be"),
        ({"name": "m", "worse": "higher", "threshold": 0.0}, "positive"),
        ({"name": "m", "worse": "higher", "threshold": float("nan")}, "positive"),
        ({"name": "m", "worse": "higher", "threshold": 1.0, "role": "blocker"}, "role must be"),
    ],
)
def test_invalid_measure_definitions(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(MeasureError, match=message):
        measure(**kwargs)


def test_missing_measure_file() -> None:
    with pytest.raises(MeasureError, match="not found"):
        load_registry(Config(measure_paths=[Path("/nonexistent/m.py")]))


def test_config_overrides_threshold_role_and_direction() -> None:
    config = Config(
        measures={
            "sparc": MeasureConfig(threshold=0.3, role="gate"),
            "min_clearance": MeasureConfig(worse="higher"),
        }
    )
    registry = load_registry(config)
    assert (registry["sparc"].threshold, registry["sparc"].role) == (0.3, "gate")
    assert registry["sparc"].worse == "lower"  # not overridden
    assert registry["min_clearance"].worse == "higher"
    assert registry["success_rate"].role == "gate"


def test_example_config_matches_builtin_defaults() -> None:
    """polcheck.toml.example documents the defaults; keep the two in step."""
    from polcheck.config import load_config

    example = load_config(Path(__file__).parents[1] / "polcheck.toml.example")
    defaults = {s.name: s for s in load_registry(Config()).specs()}
    for name, override in example.measures.items():
        if name in defaults:
            assert override.threshold == defaults[name].threshold, name
            assert override.role == defaults[name].role, name


# --- imported metrics -------------------------------------------------------------


def test_imported_metric_wins_over_builtin_and_unconfigured_metrics_are_flagged(
    store: Store,
) -> None:
    data = record(
        store,
        [
            episode(min_jerk(T), metrics={"task_time": 9.0, "task_score": 0.8, "reward": 3.0}),
            episode(min_jerk(T), metrics={"task_score": 0.7}),
        ],
    )
    config = Config(measures={"task_score": MeasureConfig(worse="lower", threshold=0.05)})
    results = compute(store, data, config)
    assert results["task_time"].origin == "imported"
    assert values(results, "task_time") == [9.0, None]  # the tool's value, not ours
    assert results["task_time"].skipped[data.runs[1].run_id] == "not in the imported data"
    assert results["task_score"].spec.configured
    assert values(results, "task_score") == [0.8, 0.7]
    assert not results["reward"].spec.configured
    assert results["sparc"].origin == "computed"


# --- caching --------------------------------------------------------------------


@pytest.fixture
def call_counter(monkeypatch: pytest.MonkeyPatch) -> Callable[[], int]:
    """Counts calls to the SPARC computation the built-in measure uses."""
    from polcheck.measures.sparc import sparc as original

    calls = 0

    def counting(*args: Any) -> float:
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr("polcheck.measures.builtin.sparc", counting)
    return lambda: calls


def test_values_are_cached_without_touching_runs_parquet(
    store: Store, call_counter: Callable[[], int]
) -> None:
    data = record(store, [episode(min_jerk(T)), episode(jerky(T))])
    runs_file = store.batch_dir(data.batch.batch_id) / "runs.parquet"
    before = runs_file.read_bytes()

    first = compute(store, data)
    assert call_counter() == 2
    second = compute(store, data)
    assert call_counter() == 2  # served from measures.parquet
    assert second["sparc"].origin == "cached"
    assert second["sparc"].values == first["sparc"].values
    assert second["task_time"].skipped == first["task_time"].skipped
    assert runs_file.read_bytes() == before


def test_version_bump_recomputes_and_drops_old_column(
    store: Store, call_counter: Callable[[], int]
) -> None:
    from dataclasses import replace

    data = record(store, [episode(min_jerk(T))])
    compute(store, data)
    registry = load_registry(Config())
    registry._specs["sparc"] = replace(registry["sparc"], version="2")
    result = measure_batch(store, data, store.get_suite("pick@1"), registry)
    assert result["sparc"].origin == "computed"
    assert call_counter() == 2
    cache = store.read_measure_cache(data.batch.batch_id)
    assert cache is not None
    assert "sparc@2" in cache.column_names
    assert "sparc@1" not in cache.column_names
    assert "task_time@1" in cache.column_names  # other measures kept


def test_results_keyed_by_run_id(store: Store) -> None:
    data = record(store, [episode(min_jerk(T))])
    results = compute(store, data)
    assert all(isinstance(k, UUID) for k in results["sparc"].values)


# --- CLI --------------------------------------------------------------------------


def run_cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["polcheck", *args])
    with pytest.raises(SystemExit) as exc:
        main()
    assert isinstance(exc.value.code, int)
    return exc.value.code


def test_measures_list_shows_builtins_files_and_imported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    custom_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "polcheck.toml").write_text(
        f'measure_paths = ["{custom_file.name}"]\n'
        '[measures.task_score]\nworse = "lower"\nthreshold = 0.05\n'
        "[measures.mystery]\nrole = 'gate'\n"
    )
    monkeypatch.chdir(tmp_path)
    assert run_cli(monkeypatch, "measures", "list") == 0
    out = capsys.readouterr().out
    for name in ("success_rate", "sparc", "task_time", "peak_wrist_force", "task_score"):
        assert name in out
    assert "successful runs only" in out
    assert "mystery" in out
    assert "unconfigured" in out


def test_measures_list_bad_file_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "broken.py").write_text("this is not python(\n")
    (tmp_path / "polcheck.toml").write_text('measure_paths = ["broken.py"]\n')
    monkeypatch.chdir(tmp_path)
    assert run_cli(monkeypatch, "measures", "list") == EXIT_USAGE
    assert "error while loading measures" in capsys.readouterr().err


def test_contacts_are_stored_when_tracked_even_if_empty(store: Store) -> None:
    data = record(store, [episode(min_jerk(T)), episode(min_jerk(T), track_contacts=False)])
    tracked, untracked = (store.read_contacts(data.batch.batch_id, r.run_id) for r in data.runs)
    assert tracked is not None
    assert tracked.num_rows == 0
    assert untracked is None


def test_rundata_contacts_property() -> None:
    data = RunData(run=None, scenario=None, ts=TimeSeries(None), contacts_table=None)  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="did not track contacts"):
        _ = data.contacts
