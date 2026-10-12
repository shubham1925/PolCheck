"""Statistics against reference values and hand-worked examples.

`fixtures/stats/reference.json` was generated with statsmodels and scipy (see
its "source" field); statsmodels is not a polcheck dependency.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from polcheck.stats import RawTest, rng_for
from polcheck.stats.continuous import paired_mean, tail_shift, unpaired_mean
from polcheck.stats.multiple import benjamini_hochberg
from polcheck.stats.sample_size import success_runs_per_side
from polcheck.stats.success import (
    mcnemar_exact,
    newcombe_interval,
    paired_success,
    unpaired_success,
    wilson_interval,
)
from polcheck.stats.verdict import judge
from tests.paths import FIXTURES

REF = json.loads((FIXTURES / "stats" / "reference.json").read_text())
B, ALPHA = 10_000, 0.05


def rng(*labels: str) -> np.random.Generator:
    return rng_for(0, *labels)


# --- McNemar ---------------------------------------------------------------------


@pytest.mark.parametrize("case", REF["mcnemar_exact"], ids=lambda c: f"b{c['b']}_c{c['c']}")
def test_mcnemar_matches_statsmodels(case: dict[str, Any]) -> None:
    assert mcnemar_exact(case["b"], case["c"]) == pytest.approx(case["p"], abs=1e-12)


def test_mcnemar_hand_worked() -> None:
    # 9 seeds flipped to failure, 1 flipped to success: 2 * P(X <= 1 | n=10, p=0.5)
    assert mcnemar_exact(9, 1) == pytest.approx(2 * (1 + 10) / 2**10)
    assert mcnemar_exact(0, 0) == 1.0  # no flips, no evidence
    assert mcnemar_exact(4, 9) == mcnemar_exact(9, 4)  # two-sided


def test_paired_success_estimate_and_interval() -> None:
    base = np.array([True] * 90 + [False] * 10)
    cand = base.copy()
    cand[:12] = False  # 12 seeds flip to failure
    cand[95:97] = True  # 2 flip to success
    result = paired_success(base, cand, rng("s"), resamples=B, alpha=ALPHA)
    assert result.estimate == pytest.approx(0.10)
    assert result.p_value == pytest.approx(mcnemar_exact(12, 2))
    assert 0.02 < result.ci_low < result.estimate < result.ci_high < 0.18


def test_paired_success_identical_outcomes() -> None:
    base = np.array([True] * 45 + [False] * 5)
    result = paired_success(base, base.copy(), rng("s"), resamples=B, alpha=ALPHA)
    assert (result.estimate, result.ci_low, result.ci_high, result.p_value) == (0.0, 0.0, 0.0, 1.0)


# --- Fisher, Wilson, Newcombe ---------------------------------------------------


@pytest.mark.parametrize("case", REF["wilson"], ids=lambda c: f"{c['k']}_of_{c['n']}")
def test_wilson_matches_statsmodels(case: dict[str, Any]) -> None:
    lo, hi = wilson_interval(case["k"], case["n"], ALPHA)
    assert lo == pytest.approx(case["lo"], abs=1e-12)
    assert hi == pytest.approx(case["hi"], abs=1e-12)


@pytest.mark.parametrize(
    "case", REF["newcombe"], ids=lambda c: f"{c['k1']}_{c['n1']}_vs_{c['k2']}_{c['n2']}"
)
def test_newcombe_matches_statsmodels(case: dict[str, Any]) -> None:
    lo, hi = newcombe_interval(case["k1"], case["n1"], case["k2"], case["n2"], ALPHA)
    assert lo == pytest.approx(case["lo"], abs=1e-9)
    assert hi == pytest.approx(case["hi"], abs=1e-9)


def test_newcombe_published_example() -> None:
    # Newcombe (1998), Statistics in Medicine 17, 873-890, Table II: 56/70 vs 48/80.
    lo, hi = newcombe_interval(56, 70, 48, 80, 0.05)
    assert (round(lo, 4), round(hi, 4)) == (0.0524, 0.3339)


@pytest.mark.parametrize("case", REF["fisher"], ids=lambda c: str(c["table"]))
def test_unpaired_success_uses_fisher(case: dict[str, Any]) -> None:
    (a, b), (c, d) = case["table"]
    result = unpaired_success(a, a + b, c, c + d, alpha=ALPHA)
    assert result.p_value == pytest.approx(case["p"], abs=1e-12)
    assert result.estimate == pytest.approx(a / (a + b) - c / (c + d))
    assert result.ci_low <= result.estimate <= result.ci_high


def test_fisher_hand_worked_tea_tasting() -> None:
    # Fisher's lady tasting tea, 3 of 4 right: P(3 or 4 right) doubled by symmetry = 34/70.
    assert unpaired_success(3, 4, 1, 4, alpha=ALPHA).p_value == pytest.approx(34 / 70)


# --- Benjamini-Hochberg ---------------------------------------------------------


@pytest.mark.parametrize("case", REF["benjamini_hochberg"], ids=lambda c: f"m{len(c['p'])}")
def test_bh_matches_statsmodels(case: dict[str, Any]) -> None:
    assert benjamini_hochberg(case["p"]) == pytest.approx(case["adjusted"], abs=1e-12)


def test_bh_hand_worked() -> None:
    # sorted: .005 .01 .03 .04 -> x4/1, x4/2, x4/3, x4/4 = .02 .02 .04 .04
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert benjamini_hochberg([]) == []
    assert benjamini_hochberg([0.2]) == [0.2]


@given(st.lists(st.floats(0, 1), min_size=1, max_size=40))
def test_bh_properties(p: list[float]) -> None:
    adjusted = benjamini_hochberg(p)
    assert all(a >= x - 1e-15 for a, x in zip(adjusted, p, strict=True))  # never smaller
    assert all(0 <= a <= 1 for a in adjusted)
    order = np.argsort(p)
    assert np.all(np.diff(np.array(adjusted)[order]) >= -1e-15)  # keeps the ordering


# --- continuous measures ----------------------------------------------------------


def test_paired_mean_detects_a_shift() -> None:
    g = np.random.default_rng(1)
    diffs = g.normal(0.5, 1.0, 100)
    result = paired_mean(diffs, rng("p"), resamples=B, alpha=ALPHA)
    assert result.estimate == pytest.approx(diffs.mean())
    assert result.ci_low > 0
    assert result.p_value < 0.001
    # BCa should be close to the textbook t interval for near-normal data.
    half = 1.984 * diffs.std(ddof=1) / 10
    assert result.ci_low == pytest.approx(diffs.mean() - half, abs=0.03)
    assert result.ci_high == pytest.approx(diffs.mean() + half, abs=0.03)


def test_paired_mean_all_zero_is_no_change() -> None:
    result = paired_mean(np.zeros(50), rng("p"), resamples=B, alpha=ALPHA)
    assert (result.estimate, result.ci_low, result.ci_high, result.p_value) == (0.0, 0.0, 0.0, 1.0)


def test_paired_mean_constant_shift() -> None:
    result = paired_mean(np.full(20, 0.3), rng("p"), resamples=B, alpha=ALPHA)
    assert (result.estimate, result.ci_low, result.ci_high) == (0.3, 0.3, 0.3)
    assert result.p_value < 0.001


def test_unpaired_mean_detects_a_shift_and_handles_constants() -> None:
    g = np.random.default_rng(2)
    base, cand = g.normal(1.0, 0.3, 80), g.normal(1.3, 0.3, 80)
    result = unpaired_mean(base, cand, rng("u"), resamples=B, alpha=ALPHA)
    assert result.estimate == pytest.approx(cand.mean() - base.mean())
    assert 0.15 < result.ci_low < result.estimate < result.ci_high < 0.45
    assert result.p_value < 1e-6

    same = unpaired_mean(np.ones(30), np.ones(30), rng("u"), resamples=B, alpha=ALPHA)
    assert (same.estimate, same.ci_low, same.ci_high, same.p_value) == (0.0, 0.0, 0.0, 1.0)


def test_tail_shift_sees_a_worse_tail_that_the_mean_misses() -> None:
    g = np.random.default_rng(3)
    base = g.normal(1.0, 0.1, 400)
    cand = base.copy()
    worst = np.argsort(cand)[-40:]  # the worst 10% get much worse
    cand[worst] += 1.0
    result = tail_shift(base, cand, rng("t"), resamples=B, alpha=ALPHA, paired=True)
    assert result.estimate > 0.5
    assert result.ci_low > 0
    assert result.p_value < 0.01

    none = tail_shift(base, base.copy(), rng("t"), resamples=B, alpha=ALPHA, paired=True)
    assert none.estimate == 0.0
    assert none.p_value == 1.0


def test_tests_are_reproducible_and_independent_of_order() -> None:
    diffs = np.random.default_rng(4).normal(0.1, 1.0, 60)
    first = paired_mean(diffs, rng("far-left", "sparc"), resamples=B, alpha=ALPHA)
    again = paired_mean(diffs, rng("far-left", "sparc"), resamples=B, alpha=ALPHA)
    other = paired_mean(diffs, rng("far-right", "sparc"), resamples=B, alpha=ALPHA)
    assert first == again
    assert first.ci_low != other.ci_low  # a different stream per test


# --- verdict rule -----------------------------------------------------------------


def raw(estimate: float, lo: float, hi: float, p: float = 0.001) -> RawTest:
    return RawTest(estimate, lo, hi, p, "test")


@pytest.mark.parametrize(
    ("test", "p_adj", "expected"),
    [
        (raw(0.10, 0.04, 0.16), 0.001, "REGRESSED"),
        (raw(0.10, 0.04, 0.16), 0.20, "INCONCLUSIVE"),  # not significant after correction
        (raw(0.10, -0.01, 0.21), 0.001, "INCONCLUSIVE"),  # interval includes zero
        (raw(0.02, 0.01, 0.029), 0.001, "PASS"),  # real but below the threshold everywhere
        (raw(0.02, 0.01, 0.05), 0.001, "INCONCLUSIVE"),  # might be above the threshold
        (raw(0.00, -0.01, 0.01), 1.0, "PASS"),
        (raw(-0.10, -0.16, -0.04), 0.001, "PASS"),  # an improvement
        (raw(0.03, 0.01, 0.05), 0.001, "REGRESSED"),  # exactly at the threshold
    ],
)
def test_verdict_rule(test: RawTest, p_adj: float, expected: str) -> None:
    assert judge(test, p_adj, 0.03, 0.05, enough_runs=True) == expected


def test_too_few_runs_is_inconclusive() -> None:
    assert judge(raw(0.5, 0.4, 0.6), 0.0001, 0.03, 0.05, enough_runs=False) == "INCONCLUSIVE"
    assert judge(None, None, 0.03, 0.05, enough_runs=True) == "INCONCLUSIVE"


@settings(max_examples=300)
@given(
    estimate=st.floats(-10, 10),
    width=st.floats(0, 10),
    p_adj=st.floats(0, 1),
    threshold=st.floats(0.001, 5),
)
def test_verdict_rule_properties(
    estimate: float, width: float, p_adj: float, threshold: float
) -> None:
    lo, hi = estimate - width, estimate + width
    verdict = judge(raw(estimate, lo, hi), p_adj, threshold, 0.05, enough_runs=True)
    if estimate <= 0:
        assert verdict != "REGRESSED"  # no change or an improvement never regresses
    if verdict == "REGRESSED":
        assert p_adj < 0.05
        assert lo > 0
        assert estimate >= threshold - 1e-9
    if verdict == "PASS":
        assert hi < threshold


# --- sample size --------------------------------------------------------------------


def test_success_sample_size() -> None:
    # Textbook two-proportion formula, 0.92 -> 0.89, alpha 0.05, power 0.8.
    p1, p2, pbar = 0.92, 0.89, 0.905
    expected = (
        1.959964 * math.sqrt(2 * pbar * (1 - pbar)) + 0.841621 * math.sqrt(p1 * 0.08 + p2 * 0.11)
    ) ** 2 / 0.03**2
    assert success_runs_per_side(0.92, 0.03) == math.ceil(expected)
    assert success_runs_per_side(0.92, 0.10) < success_runs_per_side(0.92, 0.05)
    assert success_runs_per_side(0.92, 0.05, power=0.9) > success_runs_per_side(0.92, 0.05)
    with pytest.raises(ValueError, match="drop"):
        success_runs_per_side(0.5, 0.6)
