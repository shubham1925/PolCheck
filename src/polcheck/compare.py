"""Compare two batches end to end (BUILD_PLAN 7.6).

`compare` checks comparability, computes measures, runs every test, corrects for
multiple comparisons, assigns verdicts and saves the result. `build_tests` is the
part that needs no store, so the validation harness can call it on synthetic data.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

import numpy as np

from polcheck.comparability import check
from polcheck.config import Config
from polcheck.measures import MeasureSpec, Registry, load_registry
from polcheck.measures.compute import MeasureValues, measure_batch
from polcheck.schema import (
    ALL_FAMILIES,
    BatchData,
    Comparison,
    ComparisonSettings,
    MeasureNote,
    Role,
    Run,
    TestResult,
    Verdict,
)
from polcheck.stats import RawTest, rng_for
from polcheck.stats.continuous import TAIL_QUANTILE, paired_mean, tail_shift, unpaired_mean
from polcheck.stats.multiple import benjamini_hochberg
from polcheck.stats.sample_size import success_runs_per_side
from polcheck.stats.success import paired_success, unpaired_success
from polcheck.stats.verdict import judge, overall_verdict, threshold_for
from polcheck.store import Store, StoreError

TAIL_SUFFIX = ".p95"
NOT_TESTED = "not tested: too few runs"

EXIT_PASS, EXIT_REGRESSED, EXIT_INCONCLUSIVE, EXIT_NOT_COMPARABLE = 0, 1, 2, 3


# --- Selecting runs -----------------------------------------------------------


def unique_runs(runs: Sequence[Run], deterministic: bool | None, shared: set[str]) -> list[Run]:
    """Runs in the shared scenarios. In a deterministic batch a repeated
    `(scenario, env_seed, policy_seed)` key is an exact copy (e.g. a repeat
    check), so only its first run is kept; otherwise repeats are real samples."""
    out, seen = [], set()
    for run in runs:
        if run.scenario_id not in shared:
            continue
        key = (run.scenario_id, run.env_seed, run.policy_seed)
        if deterministic and key in seen:
            continue
        seen.add(key)
        out.append(run)
    return out


def _match(base: Sequence[Run], cand: Sequence[Run]) -> list[tuple[Run, Run]]:
    """Pairs of runs with the same `(scenario, env_seed)`, first run per key."""
    first_base: dict[tuple[str, int], Run] = {}
    for run in base:
        first_base.setdefault((run.scenario_id, run.env_seed), run)
    pairs, used = [], set()
    for run in cand:
        key = (run.scenario_id, run.env_seed)
        if key in first_base and key not in used:
            used.add(key)
            pairs.append((first_base[key], run))
    return pairs


# --- Building tests -----------------------------------------------------------


@dataclass
class _Pending:
    """A test before correction and verdict."""

    family: str
    measure: str
    n_base: int
    n_cand: int
    base_value: float | None
    cand_value: float | None
    raw: RawTest | None
    threshold: float
    role: Role
    enough_runs: bool
    info_only: bool = False
    notes: list[str] = field(default_factory=list)


def _mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if len(values) else None


def _top_reason(values: MeasureValues) -> str:
    reasons = Counter(values.skipped.values())
    return reasons.most_common(1)[0][0] if reasons else "no values"


def _success_tests(
    base: Sequence[Run],
    cand: Sequence[Run],
    pairs: Sequence[tuple[Run, Run]] | None,
    families: Sequence[str],
    spec: MeasureSpec,
    config: Config,
) -> list[_Pending]:
    tests = []
    for family in [*families, ALL_FAMILIES]:

        def keep(run: Run, family: str = family) -> bool:
            return family == ALL_FAMILIES or run.family == family

        raw: RawTest | None = None
        if pairs is not None:
            b = np.array([p[0].success for p in pairs if keep(p[0])], dtype=bool)
            c = np.array([p[1].success for p in pairs if keep(p[0])], dtype=bool)
            if len(b):
                raw = paired_success(
                    b,
                    c,
                    rng_for(config.seed, family, spec.name),
                    resamples=config.bootstrap_resamples,
                    alpha=config.alpha,
                )
        else:
            b = np.array([r.success for r in base if keep(r)], dtype=bool)
            c = np.array([r.success for r in cand if keep(r)], dtype=bool)
            if len(b) and len(c):
                raw = unpaired_success(
                    int(b.sum()), len(b), int(c.sum()), len(c), alpha=config.alpha
                )
        tests.append(
            _Pending(
                family=family,
                measure=spec.name,
                n_base=len(b),
                n_cand=len(c),
                base_value=_mean(b.astype(float)),
                cand_value=_mean(c.astype(float)),
                raw=raw,
                threshold=threshold_for(spec),
                role=spec.role,
                enough_runs=min(len(b), len(c)) >= config.min_n,
            )
        )
    return tests


def _measure_tests(
    name: str,
    spec: MeasureSpec,
    base: Sequence[Run],
    cand: Sequence[Run],
    pairs: Sequence[tuple[Run, Run]] | None,
    families: Sequence[str],
    base_values: Mapping[UUID, float | None],
    cand_values: Mapping[UUID, float | None],
    success_gap: Mapping[str, float],
    success_threshold: float,
    info_note: str | None,
    config: Config,
) -> list[_Pending]:
    """Mean test per family, a pooled tail test, and per-family tail tests where
    there are enough runs. Values are oriented so that larger is worse."""
    sign = 1.0 if spec.worse == "higher" else -1.0
    threshold = threshold_for(spec)
    info_only = info_note is not None and spec.rate_sensitive
    tests = []

    def arrays(family: str) -> tuple[np.ndarray, np.ndarray]:
        """Raw values on each side; aligned pair by pair in a paired comparison."""

        def keep(run: Run) -> bool:
            return family == ALL_FAMILIES or run.family == family

        if pairs is not None:
            both = [
                (base_values.get(b.run_id), cand_values.get(c.run_id)) for b, c in pairs if keep(b)
            ]
            defined = [(x, y) for x, y in both if x is not None and y is not None]
            return (
                np.array([x for x, _ in defined], dtype=np.float64),
                np.array([y for _, y in defined], dtype=np.float64),
            )
        return (
            np.array(
                [v for r in base if keep(r) and (v := base_values.get(r.run_id)) is not None],
                dtype=np.float64,
            ),
            np.array(
                [v for r in cand if keep(r) and (v := cand_values.get(r.run_id)) is not None],
                dtype=np.float64,
            ),
        )

    def pending(family: str, measure: str, b: np.ndarray, c: np.ndarray, min_n: int) -> _Pending:
        notes = [info_note] if info_only and info_note else []
        if (
            spec.success_only
            and pairs is None
            and abs(success_gap.get(family, 0.0)) > success_threshold
        ):
            notes.append(
                "success-only measure may be biased by the change in success rate "
                "(each side's successful runs are a different mix of seeds)"
            )
        return _Pending(
            family=family,
            measure=measure,
            n_base=len(b),
            n_cand=len(c),
            base_value=None,
            cand_value=None,
            raw=None,
            threshold=threshold,
            role=spec.role,
            enough_runs=min(len(b), len(c)) >= min_n,
            info_only=info_only,
            notes=notes,
        )

    def tail_test(family: str) -> _Pending:
        b, c = arrays(family)
        test = pending(family, name + TAIL_SUFFIX, b, c, config.min_n)
        if len(b):
            test.base_value = sign * float(np.quantile(sign * b, TAIL_QUANTILE))
        if len(c):
            test.cand_value = sign * float(np.quantile(sign * c, TAIL_QUANTILE))
        if min(len(b), len(c)) >= 2:
            test.raw = tail_shift(
                sign * b,
                sign * c,
                rng_for(config.seed, family, test.measure),
                resamples=config.bootstrap_resamples,
                alpha=config.alpha,
                paired=pairs is not None,
            )
        return test

    for family in families:
        b, c = arrays(family)
        test = pending(family, name, b, c, config.min_n)
        test.base_value, test.cand_value = _mean(b), _mean(c)
        rng = rng_for(config.seed, family, name)
        if pairs is not None and len(b) >= 2:
            test.raw = paired_mean(
                sign * (c - b), rng, resamples=config.bootstrap_resamples, alpha=config.alpha
            )
        elif pairs is None and min(len(b), len(c)) >= 2:
            test.raw = unpaired_mean(
                sign * b, sign * c, rng, resamples=config.bootstrap_resamples, alpha=config.alpha
            )
        tests.append(test)
        if min(len(b), len(c)) >= config.min_n_tail:
            tests.append(tail_test(family))

    tests.append(tail_test(ALL_FAMILIES))
    return tests


def _finish(pending: Sequence[_Pending], config: Config) -> list[TestResult]:
    """Benjamini-Hochberg, verdicts and run estimates.

    Info-only tests are corrected among themselves, so they neither gain nor
    cost power for the tests that count.
    """
    adjusted: dict[int, float] = {}
    for group in (False, True):
        idx = [i for i, t in enumerate(pending) if t.info_only is group and t.raw is not None]
        p_values = [raw.p_value for i in idx if (raw := pending[i].raw) is not None]
        adjusted.update(zip(idx, benjamini_hochberg(p_values), strict=True))

    results = []
    for i, t in enumerate(pending):
        p_adjusted = adjusted.get(i)
        verdict: Verdict = judge(
            t.raw, p_adjusted, t.threshold, config.fdr_q, enough_runs=t.enough_runs
        )
        notes = list(t.notes)
        if t.raw is not None and not t.enough_runs:
            notes.append(f"fewer than min_n = {config.min_n} runs on a side")
        runs_needed = None
        if (
            verdict == "INCONCLUSIVE"
            and t.measure == "success_rate"
            and t.base_value is not None
            and t.base_value >= t.threshold
        ):
            needed = success_runs_per_side(t.base_value, t.threshold, alpha=config.alpha)
            extra = needed - min(t.n_base, t.n_cand)
            runs_needed = extra if extra > 0 else None
        results.append(
            TestResult(
                family=t.family,
                measure=t.measure,
                n_base=t.n_base,
                n_cand=t.n_cand,
                base_value=t.base_value,
                cand_value=t.cand_value,
                estimate=t.raw.estimate if t.raw else None,
                ci_low=t.raw.ci_low if t.raw else None,
                ci_high=t.raw.ci_high if t.raw else None,
                p_value=t.raw.p_value if t.raw else None,
                p_adjusted=p_adjusted,
                threshold=t.threshold,
                role=t.role,
                verdict=verdict,
                method=t.raw.method if t.raw else NOT_TESTED,
                info_only=t.info_only,
                notes=notes,
                runs_needed=runs_needed,
            )
        )
    return results


def build_tests(
    base: Sequence[Run],
    cand: Sequence[Run],
    base_measures: Mapping[str, MeasureValues],
    cand_measures: Mapping[str, MeasureValues],
    *,
    paired: bool,
    success_spec: MeasureSpec,
    config: Config,
    info_note: str | None = None,
) -> tuple[list[TestResult], list[MeasureNote], list[str]]:
    """All tests for two sets of runs (already limited to the runs to compare).

    Returns the tests, the measures that could not be tested, and notes.
    `info_note` (set for a cross-rate comparison) makes rate-sensitive measures
    info-only.
    """
    notes: list[str] = []
    pairs = _match(base, cand) if paired else None
    if pairs is not None:
        unmatched = len(base) + len(cand) - 2 * len(pairs)
        if unmatched:
            notes.append(
                f"{unmatched} run(s) had no partner with the same scenario and seed "
                "and were left out of the paired tests"
            )
    families = sorted({r.family for r in base} | {r.family for r in cand})

    pending = _success_tests(base, cand, pairs, families, success_spec, config)
    success_gap = {
        t.family: (t.base_value or 0.0) - (t.cand_value or 0.0)
        for t in pending
        if t.base_value is not None and t.cand_value is not None
    }

    not_tested: list[MeasureNote] = []
    for name in sorted(set(base_measures) | set(cand_measures)):
        b, c = base_measures.get(name), cand_measures.get(name)
        if b is None or c is None:
            side = "baseline" if b is None else "candidate"
            not_tested.append(
                MeasureNote(measure=name, status="skipped", reason=f"not in the {side} batch")
            )
            continue
        spec = c.spec
        if not spec.configured:
            not_tested.append(
                MeasureNote(
                    measure=name,
                    status="unconfigured",
                    reason=f"set worse and threshold under [measures.{name}] to compare it",
                )
            )
            continue
        empty = next((m for m in (b, c) if not m.defined()), None)
        if empty is not None:
            side = "baseline" if empty is b else "candidate"
            not_tested.append(
                MeasureNote(
                    measure=name,
                    status="skipped",
                    reason=f"no values in the {side} batch: {_top_reason(empty)}",
                )
            )
            continue
        notes.extend(f"{name}: {n}" for n in {*b.notes, *c.notes})
        pending += _measure_tests(
            name,
            spec,
            base,
            cand,
            pairs,
            families,
            b.values,
            c.values,
            success_gap,
            threshold_for(success_spec),
            info_note,
            config,
        )
    return _finish(pending, config), not_tested, notes


# --- Orchestration ------------------------------------------------------------


def resolve_batch(store: Store, ref: str, *, suite_ref: str | None = None) -> BatchData:
    """`ref` is a batch id, a policy version (its latest batch), or `baseline`
    (the current baseline for `suite_ref`)."""
    if ref == "baseline":
        if suite_ref is None:
            raise StoreError("'baseline' needs a suite; resolve the candidate first")
        batch_id = store.baseline(suite_ref)
        if batch_id is None:
            raise StoreError(f"no baseline is set for suite {suite_ref}")
        return store.read_batch(batch_id)
    try:
        return store.read_batch(UUID(ref))
    except ValueError:
        pass
    batch = store.latest_batch(ref, suite_ref) or store.latest_batch(ref)
    if batch is None:
        raise StoreError(f"no batch with id or policy version {ref!r} in {store.root}")
    return store.read_batch(batch.batch_id)


def compare(
    store: Store,
    base: BatchData,
    cand: BatchData,
    config: Config,
    *,
    assume_deterministic: bool = False,
    registry: Registry | None = None,
    save: bool = True,
) -> Comparison:
    comparability = check(base, cand, config, assume_deterministic=assume_deterministic)
    comparison_id = uuid4()
    settings = ComparisonSettings(
        alpha=config.alpha,
        fdr_q=config.fdr_q,
        min_n=config.min_n,
        min_n_tail=config.min_n_tail,
        bootstrap_resamples=config.bootstrap_resamples,
        seed=config.seed,
        assume_deterministic=assume_deterministic,
    )

    def result(
        verdict: Verdict | None,
        tests: list[TestResult],
        not_tested: list[MeasureNote],
        notes: list[str],
    ) -> Comparison:
        comparison = Comparison(
            comparison_id=comparison_id,
            created_at=datetime.now(UTC),
            baseline=base.batch,
            candidate=cand.batch,
            comparability=comparability,
            verdict=verdict,
            paired=comparability.paired,
            tests=tests,
            not_tested=not_tested,
            notes=notes,
            settings=settings,
        )
        if save:
            store.write_comparison(comparison)
        return comparison

    if comparability.status == "not_comparable":
        return result(None, [], [], [])

    registry = registry or load_registry(config)
    shared = set(comparability.shared_scenarios)
    rate = comparability.common_sample_rate_hz
    info_note = (
        None
        if rate is None
        else f"info-only: the batches were logged at different rates (compared at {rate:g} Hz)"
    )
    tests, not_tested, notes = build_tests(
        unique_runs(base.runs, comparability.base_deterministic, shared),
        unique_runs(cand.runs, comparability.cand_deterministic, shared),
        measure_batch(
            store, base, store.get_suite(base.batch.suite_ref), registry, resample_hz=rate
        ),
        measure_batch(
            store, cand, store.get_suite(cand.batch.suite_ref), registry, resample_hz=rate
        ),
        paired=comparability.paired,
        success_spec=registry["success_rate"],
        config=config,
        info_note=info_note,
    )
    return result(overall_verdict(tests), tests, not_tested, notes)


def exit_code(comparison: Comparison, *, inconclusive_exit: int = EXIT_INCONCLUSIVE) -> int:
    if comparison.verdict is None:
        return EXIT_NOT_COMPARABLE
    return {
        "PASS": EXIT_PASS,
        "REGRESSED": EXIT_REGRESSED,
        "INCONCLUSIVE": inconclusive_exit,
    }[comparison.verdict]
