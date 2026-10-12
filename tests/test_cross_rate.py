"""Comparing batches logged at different rates (DECISIONS.md D31)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from polcheck import Recorder
from polcheck.cli import main
from polcheck.compare import compare, exit_code
from polcheck.config import Config, RulesConfig
from polcheck.measures import load_registry
from polcheck.measures.compute import measure_batch
from polcheck.measures.resample import TOO_SHORT, resample_table
from polcheck.schema import BatchData
from polcheck.store import Store
from tests.paths import PICK_SUITE

NOISE = 0.0005  # 0.5 mm of position noise, like a real sensor
DURATION = 2.6
CONFIG = Config(bootstrap_resamples=1000)


def motion(t: np.ndarray) -> np.ndarray:
    """Two reaches with a pause between them, like a pick then a place."""

    def reach(s: np.ndarray) -> np.ndarray:
        s = np.clip(s, 0, 1)
        return np.asarray(10 * s**3 - 15 * s**4 + 6 * s**5)

    x = 0.25 * reach(t / 1.0) + 0.20 * reach((t - 1.4) / 1.2)
    y = 0.10 * reach((t - 1.4) / 1.2)
    z = 0.05 * np.sin(np.pi * np.clip(t / DURATION, 0, 1))
    return np.column_stack([x, y, z])


def record(
    store: Store, version: str, hz: float, *, seeds: int = 20, noise_seed: int = 0
) -> BatchData:
    """The same motion on every seed, logged at `hz` with independent sensor noise."""
    rec = Recorder(store, version, PICK_SUITE, physics_hash="ph")
    t = np.arange(0, DURATION + 1e-9, 1 / hz)
    for f, scenario in enumerate(rec.suite.scenarios):
        for seed in range(seeds):
            rng = np.random.default_rng([noise_seed, f, seed])
            pos = motion(t) + rng.normal(0, NOISE, (len(t), 3))
            with rec.run(scenario=scenario, env_seed=seed) as run:
                for k in range(len(t)):
                    run.log(t=float(t[k]), ee_pos=pos[k])
                run.finish(success=True)
    return store.read_batch(rec.close())


# --- the resampler ---------------------------------------------------------------


def table(hz: float, values: np.ndarray) -> pa.Table:
    return pa.table({"t": np.arange(len(values)) / hz, "x": values})


def rate_of(tbl: pa.Table) -> float:
    return float(1 / np.median(np.diff(tbl.column("t").to_numpy())))


@pytest.mark.parametrize(
    ("own", "target"), [(100.0, 25.0), (50.0, 25.0), (30.0, 25.0), (25.0, 25.0)]
)
def test_resampled_grid_is_at_the_target_rate(own: float, target: float) -> None:
    t = np.arange(0, 2.0, 1 / own)
    out, note = resample_table(table(own, np.sin(2 * np.pi * 1.0 * t)), target)
    assert note is None
    assert rate_of(out) == pytest.approx(target)
    assert out.column("t")[0].as_py() == 0.0
    assert out.column("t")[-1].as_py() <= t[-1] + 1e-9


def test_slow_content_is_kept_and_fast_content_removed() -> None:
    t = np.arange(0, 4.0, 0.01)  # 100 Hz
    slow, fast = np.sin(2 * np.pi * 2 * t), 0.5 * np.sin(2 * np.pi * 30 * t)
    out, _ = resample_table(table(100.0, slow + fast), 25.0)  # cutoff 10 Hz
    grid = out.column("t").to_numpy()
    middle = (grid > 0.5) & (grid < 3.5)  # away from the filter's edge effects
    error = out.column("x").to_numpy()[middle] - np.sin(2 * np.pi * 2 * grid[middle])
    assert np.abs(error).max() < 0.02  # the 2 Hz motion survives; the 30 Hz part is gone


def test_constant_signal_is_unchanged() -> None:
    out, _ = resample_table(table(100.0, np.full(200, 3.5)), 25.0)
    np.testing.assert_allclose(out.column("x").to_numpy(), 3.5)


def test_short_runs_are_interpolated_with_a_note() -> None:
    out, note = resample_table(table(100.0, np.linspace(0, 1, 12)), 25.0)
    assert note == TOO_SHORT
    assert rate_of(out) == pytest.approx(25.0)
    single, note = resample_table(table(100.0, np.array([1.0])), 25.0)
    assert note == TOO_SHORT
    assert single.num_rows == 1


# --- the acceptance check: resampling closes the gap ---------------------------------


def sparc_gap(store: Store, a: BatchData, b: BatchData, resample_hz: float | None) -> float:
    """Mean |SPARC difference| between the same seeds of two batches."""
    suite, registry = store.get_suite("pick@1"), load_registry(CONFIG)
    va = measure_batch(store, a, suite, registry, resample_hz=resample_hz)["sparc"]
    vb = measure_batch(store, b, suite, registry, resample_hz=resample_hz)["sparc"]
    gaps = [
        abs(va.values[ra.run_id] - vb.values[rb.run_id])  # type: ignore[operator]
        for ra, rb in zip(a.runs, b.runs, strict=True)
    ]
    return float(np.mean(gaps))


def test_same_motion_at_25_and_100_hz_agrees_only_after_resampling(store: Store) -> None:
    slow = record(store, "slow", 25.0, seeds=5)
    fast = record(store, "fast", 100.0, seeds=5, noise_seed=1)
    raw = sparc_gap(store, slow, fast, None)
    resampled = sparc_gap(store, slow, fast, 25.0)
    assert raw > 0.1  # more than SPARC's threshold from the logging rate alone
    assert resampled < 0.1
    assert resampled < raw / 3


def test_resampled_values_have_their_own_cache_key(store: Store) -> None:
    fast = record(store, "fast", 100.0, seeds=2)
    suite, registry = store.get_suite("pick@1"), load_registry(CONFIG)
    full = measure_batch(store, fast, suite, registry)
    resampled = measure_batch(store, fast, suite, registry, resample_hz=25.0)
    cache = store.read_measure_cache(fast.batch.batch_id)
    assert cache is not None
    assert {"sparc@1", "sparc@1@25Hz", "task_time@1"} <= set(cache.column_names)
    assert "task_time@1@25Hz" not in cache.column_names  # not rate-sensitive
    assert full["sparc"].values != resampled["sparc"].values
    assert resampled["sparc"].resampled_hz == 25.0
    assert full["task_time"].values == resampled["task_time"].values
    # Both sets are served from the cache afterwards.
    assert measure_batch(store, fast, suite, registry)["sparc"].origin == "cached"
    again = measure_batch(store, fast, suite, registry, resample_hz=25.0)["sparc"]
    assert (again.origin, again.values) == ("cached", resampled["sparc"].values)


# --- the comparison ------------------------------------------------------------------


def test_cross_rate_comparison_is_info_only_for_rate_sensitive_measures(store: Store) -> None:
    slow = record(store, "slow", 25.0)
    fast = record(store, "fast", 100.0, noise_seed=1)
    result = compare(store, fast, slow, CONFIG, assume_deterministic=True)

    assert result.comparability.common_sample_rate_hz == pytest.approx(25.0)
    (issue,) = [i for i in result.comparability.issues if i.rule == "sample_rate"]
    assert issue.severity == "warn"
    assert "resampling both sides to 25 Hz" in issue.message
    assert "info-only" in issue.message

    by_measure = {t.measure.removesuffix(".p95"): t.info_only for t in result.tests}
    assert by_measure == {
        "success_rate": False,
        "task_time": False,  # a duration barely depends on the rate
        "sparc": True,
        "log_dimensionless_jerk": True,
        "hesitation_time": True,
    }
    info = [t for t in result.tests if t.info_only]
    assert all("logged at different rates (compared at 25 Hz)" in t.notes[0] for t in info)
    assert all(t.estimate is not None for t in info)

    # Jerk stays rate-dependent even after resampling: it "regresses" with no change
    # in behaviour. As an info-only test it is shown but cannot fail the comparison.
    assert "REGRESSED" in {t.verdict for t in info}
    assert result.verdict == "PASS"
    assert exit_code(result) == 0


def test_cross_rate_cli_labels_info_only_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = Store(tmp_path / ".polcheck")
    record(store, "slow", 25.0)
    record(store, "fast", 100.0, noise_seed=1)
    (tmp_path / "polcheck.toml").write_text("bootstrap_resamples = 1000\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "polcheck",
            "compare",
            "--baseline",
            "fast",
            "--candidate",
            "slow",
            "--assume-deterministic",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    out = capsys.readouterr().out
    assert exc.value.code == 0
    assert out.startswith("PASS")
    assert "REGRESSED (info)" in out
    assert "warn: [sample_rate]" in out


def test_non_integer_rate_ratio_works(store: Store) -> None:
    a = record(store, "a", 25.0)
    b = record(store, "b", 30.0, noise_seed=1)
    result = compare(store, a, b, CONFIG, assume_deterministic=True)
    assert result.comparability.common_sample_rate_hz == pytest.approx(25.0)
    sparc = [t for t in result.tests if t.measure == "sparc"]
    assert len(sparc) == 4
    assert all(t.info_only and t.estimate is not None for t in sparc)


def test_same_rate_comparison_has_no_info_only_tests(store: Store) -> None:
    a = record(store, "a", 25.0)
    b = record(store, "b", 25.0, noise_seed=1)
    result = compare(store, a, b, CONFIG, assume_deterministic=True)
    assert result.comparability.common_sample_rate_hz is None
    assert not any(t.info_only for t in result.tests)
    cache = store.read_measure_cache(a.batch.batch_id)
    assert cache is not None
    assert not any("Hz" in name for name in cache.column_names)


def test_sample_rate_rule_can_still_block(store: Store) -> None:
    a = record(store, "a", 25.0, seeds=2)
    b = record(store, "b", 100.0, seeds=2)
    strict = CONFIG.model_copy(update={"rules": RulesConfig(sample_rate="block")})
    result = compare(store, a, b, strict, assume_deterministic=True)
    assert result.verdict is None
    assert result.comparability.common_sample_rate_hz is None
    assert exit_code(result) == 3
