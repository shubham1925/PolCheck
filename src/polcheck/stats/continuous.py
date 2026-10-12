"""Tests for continuous measures: mean shift (paired or unpaired) and tail shift.

Inputs are already oriented so that larger values are worse; every estimate is
"candidate minus baseline", so positive means the candidate is worse.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from polcheck.stats import RawTest

FloatArray = NDArray[np.float64]

TAIL_QUANTILE = 0.95
_CHUNK = 1000  # bootstrap resamples per chunk, to bound memory on large batches


def _bca_interval(
    samples: tuple[FloatArray, ...],
    statistic: object,
    rng: np.random.Generator,
    *,
    resamples: int,
    alpha: float,
    paired: bool,
) -> tuple[float, float]:
    """BCa bootstrap interval, falling back to the percentile method when the
    acceleration term is undefined (e.g. many tied values)."""
    for method in ("BCa", "percentile"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = stats.bootstrap(
                samples,
                statistic,
                n_resamples=resamples,
                confidence_level=1 - alpha,
                method=method,
                paired=paired,
                vectorized=True,
                rng=rng,
            )
        lo, hi = float(result.confidence_interval.low), float(result.confidence_interval.high)
        if np.isfinite(lo) and np.isfinite(hi):
            return lo, hi
    raise ArithmeticError("bootstrap interval is not finite")  # unreachable for finite input


def _mean(x: FloatArray, axis: int = -1) -> FloatArray:
    return np.asarray(np.mean(x, axis=axis))


def _mean_difference(a: FloatArray, b: FloatArray, axis: int = -1) -> FloatArray:
    return np.asarray(np.mean(a, axis=axis) - np.mean(b, axis=axis))


def paired_mean(
    diffs: FloatArray, rng: np.random.Generator, *, resamples: int, alpha: float
) -> RawTest:
    """Mean of per-seed differences (candidate - baseline). Needs at least 2 pairs.

    CI by BCa bootstrap; p from the Wilcoxon signed-rank test. If every
    difference is identical there is nothing to resample: the interval collapses
    to that value, and all-zero differences mean "no change" (p = 1).
    """
    diffs = np.asarray(diffs, dtype=np.float64)
    estimate = float(diffs.mean())
    if np.ptp(diffs) == 0:
        estimate = float(diffs[0])  # exact, not the rounded mean of identical values
        ci = (estimate, estimate)
        p_value = 1.0 if estimate == 0 else float(stats.wilcoxon(diffs).pvalue)
    else:
        ci = _bca_interval((diffs,), _mean, rng, resamples=resamples, alpha=alpha, paired=False)
        p_value = float(stats.wilcoxon(diffs).pvalue)
    return RawTest(estimate, ci[0], ci[1], p_value, "Wilcoxon signed-rank; BCa bootstrap CI")


def unpaired_mean(
    base: FloatArray,
    cand: FloatArray,
    rng: np.random.Generator,
    *,
    resamples: int,
    alpha: float,
) -> RawTest:
    """Difference in means, candidate - baseline. Needs at least 2 values per side.

    CI by two-sample BCa bootstrap; p from the Mann-Whitney U test.
    """
    base, cand = np.asarray(base, dtype=np.float64), np.asarray(cand, dtype=np.float64)
    estimate = float(cand.mean() - base.mean())
    if np.ptp(base) == 0 and np.ptp(cand) == 0:
        ci = (estimate, estimate)
    else:
        ci = _bca_interval(
            (cand, base), _mean_difference, rng, resamples=resamples, alpha=alpha, paired=False
        )
    if estimate == 0 and np.ptp(base) == 0 and np.ptp(cand) == 0:
        p_value = 1.0
    else:
        p_value = float(stats.mannwhitneyu(cand, base, alternative="two-sided").pvalue)
    return RawTest(estimate, ci[0], ci[1], p_value, "Mann-Whitney U; BCa bootstrap CI")


def tail_shift(
    base: FloatArray,
    cand: FloatArray,
    rng: np.random.Generator,
    *,
    resamples: int,
    alpha: float,
    paired: bool,
    quantile: float = TAIL_QUANTILE,
) -> RawTest:
    """Difference in the 95th percentile (the bad tail), candidate - baseline.

    Percentile bootstrap. `paired` resamples seeds jointly (the arrays must be
    aligned); otherwise each side is resampled independently. There is no
    classical test for a quantile difference, so the p-value comes from the
    bootstrap distribution: twice the smaller share of resampled differences on
    either side of zero.
    """
    base, cand = np.asarray(base, dtype=np.float64), np.asarray(cand, dtype=np.float64)
    estimate = float(np.quantile(cand, quantile) - np.quantile(base, quantile))
    boot = np.empty(resamples)
    for start in range(0, resamples, _CHUNK):
        size = min(_CHUNK, resamples - start)
        idx_c = rng.integers(0, len(cand), (size, len(cand)))
        idx_b = idx_c if paired else rng.integers(0, len(base), (size, len(base)))
        boot[start : start + size] = np.quantile(cand[idx_c], quantile, axis=1) - np.quantile(
            base[idx_b], quantile, axis=1
        )
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    at_or_below = (np.sum(boot <= 0) + 1) / (resamples + 1)
    at_or_above = (np.sum(boot >= 0) + 1) / (resamples + 1)
    p_value = float(min(1.0, 2 * min(at_or_below, at_or_above)))
    return RawTest(estimate, float(lo), float(hi), p_value, "percentile bootstrap (p95 difference)")
