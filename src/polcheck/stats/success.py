"""Success-rate tests: McNemar (paired) and Fisher with Newcombe's interval (unpaired)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from polcheck.stats import RawTest

BoolArray = NDArray[np.bool_]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts.

    `b` = baseline succeeded and candidate failed; `c` = the reverse. Under "no
    change" each discordant pair is equally likely to go either way, so `b` is
    Binomial(b + c, 0.5). With no discordant pairs there is no evidence: p = 1.
    """
    if b + c == 0:
        return 1.0
    return float(stats.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)


def paired_success(
    base: BoolArray,
    cand: BoolArray,
    rng: np.random.Generator,
    *,
    resamples: int,
    alpha: float,
) -> RawTest:
    """Paired comparison of success on the same seeds.

    Estimate = (b - c) / n, the drop in success rate (positive = worse). The
    interval is a percentile bootstrap over seeds. Resampling n pairs with
    replacement only changes how many fall in each of three groups (b, c, same
    outcome), so the bootstrap draws those counts from a multinomial directly.
    """
    base, cand = np.asarray(base, dtype=bool), np.asarray(cand, dtype=bool)
    n = len(base)
    b = int(np.sum(base & ~cand))
    c = int(np.sum(~base & cand))
    counts = rng.multinomial(n, [b / n, c / n, (n - b - c) / n], size=resamples)
    boot = (counts[:, 0] - counts[:, 1]) / n
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return RawTest(
        estimate=(b - c) / n,
        ci_low=float(lo),
        ci_high=float(hi),
        p_value=mcnemar_exact(b, c),
        method="McNemar exact; paired bootstrap CI",
    )


def wilson_interval(k: int, n: int, alpha: float) -> tuple[float, float]:
    """Wilson score interval for a proportion (no continuity correction)."""
    z = float(stats.norm.ppf(1 - alpha / 2))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_interval(k1: int, n1: int, k2: int, n2: int, alpha: float) -> tuple[float, float]:
    """Newcombe's hybrid score interval for p1 - p2 (his 1998 "method 10").

    Combines the two Wilson intervals; unlike the textbook Wald interval it
    behaves at rates near 0 or 1, where success rates usually are.
    """
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_interval(k1, n1, alpha)
    l2, u2 = wilson_interval(k2, n2, alpha)
    diff = p1 - p2
    lo = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lo, hi


def unpaired_success(
    base_k: int, base_n: int, cand_k: int, cand_n: int, *, alpha: float
) -> RawTest:
    """Unpaired comparison of two success counts.

    Estimate = baseline rate - candidate rate (positive = worse).
    """
    table = [[base_k, base_n - base_k], [cand_k, cand_n - cand_k]]
    p_value = float(stats.fisher_exact(table, alternative="two-sided")[1])
    lo, hi = newcombe_interval(base_k, base_n, cand_k, cand_n, alpha)
    return RawTest(
        estimate=base_k / base_n - cand_k / cand_n,
        ci_low=lo,
        ci_high=hi,
        p_value=p_value,
        method="Fisher exact; Newcombe score CI",
    )
