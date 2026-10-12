from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polcheck.cli import EXIT_USAGE, main
from polcheck.compare import build_tests, compare, exit_code, resolve_batch, unique_runs
from polcheck.config import Config, MeasureConfig, RulesConfig
from polcheck.measures import load_registry
from polcheck.measures.compute import measure_batch
from polcheck.schema import (
    ALL_FAMILIES,
    Batch,
    BatchData,
    Comparison,
    Run,
    SimulatorInfo,
    TestResult,
)
from polcheck.stats.verdict import overall_verdict
from polcheck.store import Store, StoreError
from polcheck.suite import family_scenarios, load_suite
from tests.paths import PICK_SUITE

T0 = datetime(2026, 10, 9, tzinfo=UTC)
FAMILIES = ["near-left", "near-right", "far-left", "far-right"]
SCENARIOS = {f: ids[0] for f, ids in family_scenarios(load_suite(PICK_SUITE)).items()}

# A fast config: fewer bootstrap resamples, and one imported metric that counts.
CONFIG = Config(
    bootstrap_resamples=2000,
    measures={
        "score": MeasureConfig(worse="lower", threshold=0.05),
        "effort": MeasureConfig(worse="higher", threshold=1.0),
    },
)

RunSpec = tuple[bool, dict[str, float]]
"""(success, metrics) for one seed."""


def make_batch(
    store: Store,
    version: str,
    families: dict[str, list[RunSpec]],
    *,
    deterministic: bool | None = True,
    physics_hash: str = "ph",
    age: int = 0,
) -> BatchData:
    store.register_suite(PICK_SUITE)
    batch_id = uuid4()
    runs = [
        Run(
            run_id=uuid4(),
            batch_id=batch_id,
            policy_version=version,
            suite_ref="pick@1",
            scenario_id=SCENARIOS[family],
            family=family,
            env_seed=seed,
            simulator=SimulatorInfo(name="sim", version="1"),
            physics_hash=physics_hash,
            success=success,
            metrics=metrics,
            source="tabular",
            has_timeseries=False,
            started_at=T0,
        )
        for family, specs in families.items()
        for seed, (success, metrics) in enumerate(specs)
    ]
    batch = Batch(
        batch_id=batch_id,
        policy_version=version,
        suite_ref="pick@1",
        created_at=T0 + timedelta(minutes=age),
        deterministic=deterministic,
        n_runs=len(runs),
    )
    data = BatchData(batch=batch, runs=runs)
    store.write_batch(data)
    return data


def as_runs(family: str, specs: list[tuple[bool, float]], batch_id: Any) -> list[Run]:
    return [
        Run(
            run_id=uuid4(),
            batch_id=batch_id,
            policy_version="v",
            suite_ref="pick@1",
            scenario_id=SCENARIOS[family],
            family=family,
            env_seed=seed,
            simulator=SimulatorInfo(name="sim", version="1"),
            physics_hash="ph",
            success=success,
            metrics={"score": score},
            source="tabular",
            has_timeseries=False,
            started_at=T0,
        )
        for seed, (success, score) in enumerate(specs)
    ]


def family_runs(n: int, successes: int, seed: int, *, score: float = 0.8) -> list[RunSpec]:
    """`successes` of `n` seeds succeed (the first ones); metrics vary by seed."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(score, 0.05, n)
    return [(i < successes, {"score": float(scores[i])}) for i in range(n)]


def standard(seed: int = 0, n: int = 50) -> dict[str, list[RunSpec]]:
    return {f: family_runs(n, round(0.96 * n), seed + i) for i, f in enumerate(FAMILIES)}


def find(comparison: Comparison, family: str, measure: str) -> TestResult:
    (test,) = [t for t in comparison.tests if (t.family, t.measure) == (family, measure)]
    return test


# --- verdicts end to end ----------------------------------------------------------


def test_identical_batches_pass_and_are_saved(store: Store) -> None:
    base = make_batch(store, "v1", standard())
    cand = make_batch(store, "v2", standard(), age=1)
    result = compare(store, base, cand, CONFIG)
    assert result.verdict == "PASS"
    assert exit_code(result) == 0
    assert result.paired
    assert {t.verdict for t in result.tests} == {"PASS"}
    assert all(t.estimate == 0 and t.p_value == 1.0 for t in result.tests)
    # 5 success tests + 4 score means + 1 pooled score tail
    assert len(result.tests) == 10
    assert find(result, "far-left", "success_rate").method.startswith("McNemar")
    assert store.read_comparison(result.comparison_id) == result
    assert store.query("SELECT verdict, candidate_policy FROM comparisons") == [("PASS", "v2")]


def test_success_drop_in_one_family_regresses(store: Store) -> None:
    families = standard()
    base = make_batch(store, "v1", families)
    worse = dict(families)
    worse["far-left"] = family_runs(50, 30, 2)  # 48 -> 30 successes
    cand = make_batch(store, "v2", worse, age=1)
    result = compare(store, base, cand, CONFIG)
    assert result.verdict == "REGRESSED"
    assert exit_code(result) == 1
    far = find(result, "far-left", "success_rate")
    assert far.verdict == "REGRESSED"
    assert far.estimate == pytest.approx(18 / 50)
    assert (far.base_value, far.cand_value) == (0.96, 0.6)
    assert find(result, ALL_FAMILIES, "success_rate").verdict == "REGRESSED"
    assert find(result, "near-left", "success_rate").verdict == "PASS"
    assert all(t.p_adjusted >= t.p_value for t in result.tests if t.p_value is not None)  # type: ignore[operator]


def test_inconclusive_gate_makes_the_comparison_inconclusive(store: Store) -> None:
    small = {f: family_runs(10, 9, i) for i, f in enumerate(FAMILIES)}  # below min_n = 20
    base = make_batch(store, "v1", small)
    cand = make_batch(store, "v2", small, age=1)
    result = compare(store, base, cand, CONFIG)
    assert result.verdict == "INCONCLUSIVE"
    assert exit_code(result) == 2
    assert exit_code(result, inconclusive_exit=0) == 0
    test = find(result, "near-left", "success_rate")
    assert test.verdict == "INCONCLUSIVE"
    assert "fewer than min_n" in test.notes[0]
    assert test.runs_needed is not None
    assert test.runs_needed > 100
    assert result.comparability.low_n_families == sorted(FAMILIES)


def noisy_metric(n: int, seed: int, shift: float) -> list[RunSpec]:
    rng = np.random.default_rng(seed)
    return [(True, {"score": 0.8, "effort": float(v)}) for v in rng.normal(10 + shift, 4.0, n)]


def test_inconclusive_advisory_measure_does_not_block(store: Store) -> None:
    base = make_batch(store, "v1", {f: noisy_metric(50, i, 0.0) for i, f in enumerate(FAMILIES)})
    cand = make_batch(
        store, "v2", {f: noisy_metric(50, 100 + i, 0.3) for i, f in enumerate(FAMILIES)}, age=1
    )
    result = compare(store, base, cand, CONFIG)
    effort = [t for t in result.tests if t.measure.startswith("effort")]
    assert "INCONCLUSIVE" in {t.verdict for t in effort}  # too noisy to settle at threshold 1.0
    assert "REGRESSED" not in {t.verdict for t in effort}
    assert all(t.role == "advisory" for t in effort)
    assert result.verdict == "PASS"


def test_advisory_regression_regresses_the_comparison(store: Store) -> None:
    base = make_batch(store, "v1", {f: noisy_metric(50, i, 0.0) for i, f in enumerate(FAMILIES)})
    cand = make_batch(
        store, "v2", {f: noisy_metric(50, 100 + i, 6.0) for i, f in enumerate(FAMILIES)}, age=1
    )
    result = compare(store, base, cand, CONFIG)
    assert result.verdict == "REGRESSED"
    test = find(result, "far-right", "effort")
    assert test.verdict == "REGRESSED"
    assert test.estimate == pytest.approx(test.cand_value - test.base_value)  # type: ignore[operator]


def test_lower_is_worse_measures_are_oriented(store: Store) -> None:
    base = make_batch(
        store, "v1", {f: family_runs(50, 48, i, score=0.8) for i, f in enumerate(FAMILIES)}
    )
    cand = make_batch(
        store, "v2", {f: family_runs(50, 48, i, score=0.6) for i, f in enumerate(FAMILIES)}, age=1
    )
    result = compare(store, base, cand, CONFIG)
    test = find(result, "near-left", "score")
    assert test.estimate == pytest.approx(0.2)  # a drop of 0.2 in a lower-is-worse measure
    assert test.verdict == "REGRESSED"
    # The bad tail of a lower-is-worse measure is its low end.
    tail = find(result, ALL_FAMILIES, "score.p95")
    assert tail.base_value is not None
    assert tail.base_value < 0.8


def test_not_comparable_stops_with_exit_3(store: Store) -> None:
    base = make_batch(store, "v1", standard())
    cand = make_batch(store, "v2", standard(), physics_hash="other", age=1)
    result = compare(store, base, cand, CONFIG)
    assert result.verdict is None
    assert result.tests == []
    assert exit_code(result) == 3
    assert store.read_comparison(result.comparison_id).comparability.status == "not_comparable"


def test_unpaired_comparison_uses_fisher_and_mann_whitney(store: Store) -> None:
    base = make_batch(store, "v1", standard(), deterministic=None)
    cand = make_batch(store, "v2", standard(seed=50), deterministic=None, age=1)
    result = compare(store, base, cand, CONFIG)
    assert not result.paired
    assert find(result, "far-left", "success_rate").method.startswith("Fisher")
    assert find(result, "far-left", "score").method.startswith("Mann-Whitney")
    assert compare(store, base, cand, CONFIG, assume_deterministic=True).paired


# --- success-only measures ---------------------------------------------------------


def survivorship_batches(store: Store, deterministic: bool | None) -> tuple[BatchData, BatchData]:
    """The candidate is identical except that it fails the 10 slowest seeds."""
    times = np.linspace(1.0, 3.0, 50)
    base_runs = [(True, {"task_time": float(t)}) for t in times]
    cand_runs = [(t < times[40], {"task_time": float(t)}) for t in times]
    base = make_batch(store, "v1", {f: base_runs for f in FAMILIES}, deterministic=deterministic)
    cand = make_batch(
        store, "v2", {f: cand_runs for f in FAMILIES}, deterministic=deterministic, age=1
    )
    return base, cand


def test_success_only_measure_uses_seeds_where_both_succeeded(store: Store) -> None:
    base, cand = survivorship_batches(store, deterministic=True)
    result = compare(store, base, cand, CONFIG)
    test = find(result, "far-left", "task_time")
    assert (test.n_base, test.n_cand) == (40, 40)
    assert test.estimate == 0.0  # same seeds, same times: it must not look faster
    assert test.verdict == "PASS"
    assert find(result, "far-left", "success_rate").verdict == "REGRESSED"


def test_unpaired_success_only_measure_warns_about_bias(store: Store) -> None:
    base, cand = survivorship_batches(store, deterministic=None)
    result = compare(store, base, cand, CONFIG)
    test = find(result, "far-left", "task_time")
    assert (test.n_base, test.n_cand) == (50, 40)
    assert test.estimate is not None
    assert test.estimate < 0  # looks faster only because the slow seeds now fail
    assert any("may be biased by the change in success rate" in n for n in test.notes)
    assert test.verdict != "REGRESSED"


# --- tail tests and repeats ----------------------------------------------------------


def test_tail_tests_pooled_always_per_family_only_with_enough_runs(store: Store) -> None:
    sizes = {"near-left": 60, "near-right": 60, "far-left": 25, "far-right": 25}
    families = {f: family_runs(n, n, i) for i, (f, n) in enumerate(sizes.items())}
    base = make_batch(store, "v1", families)
    cand = make_batch(store, "v2", families, age=1)
    config = CONFIG.model_copy(update={"min_n_tail": 50})
    tails = {t.family for t in compare(store, base, cand, config).tests if t.measure == "score.p95"}
    assert tails == {ALL_FAMILIES, "near-left", "near-right"}
    default_tails = {
        t.family for t in compare(store, base, cand, CONFIG).tests if t.measure == "score.p95"
    }
    assert default_tails == {ALL_FAMILIES}  # min_n_tail = 100


def test_deterministic_repeats_are_counted_once() -> None:
    batch_id = uuid4()
    runs = as_runs("far-left", [(True, 0.8)] * 50, batch_id)
    runs += as_runs("near-left", [(True, 0.8)] * 50, batch_id)
    repeats = as_runs("far-left", [(True, 0.8)] * 2, batch_id)  # seeds 0 and 1 again
    shared = set(SCENARIOS.values())
    assert len(unique_runs(runs + repeats, True, shared)) == 100  # exact copies dropped
    assert len(unique_runs(runs + repeats, False, shared)) == 102  # real samples kept
    assert len(unique_runs(runs + repeats, None, shared)) == 102
    assert len(unique_runs(runs + repeats, True, {SCENARIOS["far-left"]})) == 50


# --- measures that cannot be tested ---------------------------------------------------


def test_unconfigured_and_missing_measures_are_reported_not_tested(store: Store) -> None:
    def runs(extra: dict[str, float]) -> dict[str, list[RunSpec]]:
        return {f: [(True, {"score": 0.8, **extra})] * 30 for f in FAMILIES}

    base = make_batch(store, "v1", runs({"reward": 1.0, "legacy": 2.0}), deterministic=None)
    cand = make_batch(store, "v2", runs({"reward": 1.0}), deterministic=None, age=1)
    result = compare(store, base, cand, CONFIG)
    notes = {n.measure: (n.status, n.reason) for n in result.not_tested}
    assert notes["reward"][0] == "unconfigured"
    assert "[measures.reward]" in notes["reward"][1]
    assert notes["legacy"] == ("skipped", "not in the candidate batch")
    # Built-ins need time-series these batches do not have.
    assert notes["sparc"][0] == "skipped"
    assert "missing signal(s)" in notes["sparc"][1]
    assert {t.measure for t in result.tests} == {"success_rate", "score", "score.p95"}


# --- properties (BUILD_PLAN M4 acceptance) ----------------------------------------------

FAST = CONFIG.model_copy(update={"bootstrap_resamples": 200})
runs_strategy = st.lists(st.tuples(st.booleans(), st.floats(0.0, 1.0)), min_size=20, max_size=40)


def run_build(base: list[Run], cand: list[Run], paired: bool) -> list[TestResult]:
    from polcheck.measures.compute import _imported

    registry = load_registry(FAST)
    (spec,) = registry.imported({"score"})
    tests, _, _ = build_tests(
        base,
        cand,
        {"score": _imported(spec, base)},
        {"score": _imported(spec, cand)},
        paired=paired,
        success_spec=registry["success_rate"],
        config=FAST,
    )
    return tests


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(specs=runs_strategy, paired=st.booleans())
def test_identical_batches_never_regress(specs: list[tuple[bool, float]], paired: bool) -> None:
    base = as_runs("far-left", specs, uuid4())
    cand = as_runs("far-left", specs, uuid4())
    tests = run_build(base, cand, paired)
    assert "REGRESSED" not in {t.verdict for t in tests}
    assert overall_verdict(tests) != "REGRESSED"


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    specs=runs_strategy,
    gains=st.lists(st.floats(0.0, 0.5), min_size=40, max_size=40),
    fixes=st.lists(st.booleans(), min_size=40, max_size=40),
    paired=st.booleans(),
)
def test_improvements_never_regress(
    specs: list[tuple[bool, float]], gains: list[float], fixes: list[bool], paired: bool
) -> None:
    # Candidate: every seed's score is at least as high, and failures may become successes.
    better = [
        (success or fix, score + gain)
        for (success, score), gain, fix in zip(specs, gains, fixes, strict=False)
    ]
    tests = run_build(
        as_runs("far-left", specs, uuid4()), as_runs("far-left", better, uuid4()), paired
    )
    assert "REGRESSED" not in {t.verdict for t in tests}


verdicts = st.sampled_from(["PASS", "REGRESSED", "INCONCLUSIVE"])
roles = st.sampled_from(["gate", "advisory"])


def judged(verdict: str, role: str, info_only: bool) -> TestResult:
    return TestResult(
        family="f",
        measure="m",
        n_base=1,
        n_cand=1,
        base_value=None,
        cand_value=None,
        estimate=None,
        ci_low=None,
        ci_high=None,
        p_value=None,
        p_adjusted=None,
        threshold=1.0,
        role=role,  # type: ignore[arg-type]
        verdict=verdict,  # type: ignore[arg-type]
        method="x",
        info_only=info_only,
    )


@given(st.lists(st.tuples(verdicts, roles, st.booleans()), max_size=12))
def test_rollup_matches_the_rule(items: list[tuple[str, str, bool]]) -> None:
    counted = [(v, r) for v, r, info in items if not info]
    if any(v == "REGRESSED" for v, _ in counted):
        expected = "REGRESSED"
    elif any(v == "INCONCLUSIVE" and r == "gate" for v, r in counted):
        expected = "INCONCLUSIVE"
    else:
        expected = "PASS"
    assert overall_verdict([judged(*item) for item in items]) == expected


@given(st.lists(st.tuples(st.sampled_from(["PASS", "INCONCLUSIVE"]), st.booleans()), max_size=12))
def test_inconclusive_advisory_never_makes_the_overall_inconclusive(
    items: list[tuple[str, bool]],
) -> None:
    tests = [judged(verdict, "advisory", info) for verdict, info in items]
    assert overall_verdict(tests) == "PASS"


def test_info_only_regression_does_not_count() -> None:
    assert (
        overall_verdict([judged("REGRESSED", "gate", True), judged("PASS", "gate", False)])
        == "PASS"
    )


# --- resolving batches and the CLI -------------------------------------------------------


def test_resolve_batch(store: Store) -> None:
    old = make_batch(store, "v1", standard())
    new = make_batch(store, "v1", standard(), age=5)
    assert resolve_batch(store, str(old.batch.batch_id)).batch == old.batch
    assert resolve_batch(store, "v1").batch == new.batch  # latest batch of that version
    with pytest.raises(StoreError, match="no baseline is set"):
        resolve_batch(store, "baseline", suite_ref="pick@1")
    store.set_baseline(old.batch.batch_id)
    assert resolve_batch(store, "baseline", suite_ref="pick@1").batch == old.batch
    with pytest.raises(StoreError, match="no batch with id or policy version 'nope'"):
        resolve_batch(store, "nope")


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., int]:
    (tmp_path / "polcheck.toml").write_text(
        'bootstrap_resamples = 1000\n[measures.score]\nworse = "lower"\nthreshold = 0.05\n'
    )
    monkeypatch.chdir(tmp_path)

    def run(*args: str) -> int:
        monkeypatch.setattr(sys, "argv", ["polcheck", *args])
        with pytest.raises(SystemExit) as exc:
            main()
        assert isinstance(exc.value.code, int)
        return exc.value.code

    return run


def test_compare_command_exit_codes(
    cli: Callable[..., int], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = Store(tmp_path / ".polcheck")
    families = standard()
    make_batch(store, "v1", families)
    make_batch(store, "same", families, age=1)
    worse = dict(families)
    worse["far-left"] = family_runs(50, 30, 2)
    make_batch(store, "worse", worse, age=2)
    make_batch(store, "tiny", {f: family_runs(10, 9, i) for i, f in enumerate(FAMILIES)}, age=3)
    make_batch(store, "other-physics", families, physics_hash="x", age=4)

    assert cli("compare", "--baseline", "v1", "--candidate", "same") == 0
    out = capsys.readouterr().out
    assert out.startswith("PASS: candidate same vs baseline v1 on pick@1")
    assert "Saved as comparison" in out

    assert cli("compare", "--baseline", "v1", "--candidate", "worse") == 1
    out = capsys.readouterr().out
    assert out.startswith("REGRESSED")
    assert "far-left" in out.splitlines()[4]  # the worst test is listed first

    assert cli("compare", "--baseline", "tiny", "--candidate", "tiny") == 2
    assert "more runs per side" in capsys.readouterr().out
    assert (
        cli("compare", "--baseline", "tiny", "--candidate", "tiny", "--inconclusive-exit", "0") == 0
    )
    assert cli("compare", "--baseline", "v1", "--candidate", "other-physics") == 3
    assert capsys.readouterr().out.count("NOT COMPARABLE") == 1
    assert cli("compare", "--baseline", "v1", "--candidate", "missing") == EXIT_USAGE
    assert (
        cli("compare", "--baseline", "v1", "--candidate", "same", "--inconclusive-exit", "1")
        == EXIT_USAGE
    )


def test_comparisons_survive_reindex(store: Store) -> None:
    base = make_batch(store, "v1", standard())
    cand = make_batch(store, "v2", standard(), age=1)
    result = compare(store, base, cand, CONFIG)
    before = store.query("SELECT * FROM comparisons")
    store.index_path.unlink()
    store.reindex()
    assert store.query("SELECT * FROM comparisons") == before
    assert before[0][0] == str(result.comparison_id)


def test_block_rule_from_config(store: Store) -> None:
    base = make_batch(store, "v1", standard())
    cand = make_batch(store, "v2", {"near-left": standard()["near-left"]}, age=1)
    blocking = CONFIG.model_copy(update={"rules": RulesConfig(scenario_set="block")})
    assert compare(store, base, cand, blocking).verdict is None
    partial = compare(store, base, cand, CONFIG)
    assert partial.comparability.status == "partial"
    assert {t.family for t in partial.tests} == {"near-left", ALL_FAMILIES}


def test_measure_cache_is_reused_across_comparisons(store: Store) -> None:
    base = make_batch(store, "v1", standard())
    suite = store.get_suite("pick@1")
    first = measure_batch(store, base, suite, load_registry(CONFIG))
    second = measure_batch(store, base, suite, load_registry(CONFIG))
    assert first["score"].values == second["score"].values
