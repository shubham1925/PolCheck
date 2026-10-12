"""How many runs a comparison needs."""

from __future__ import annotations

import math

from scipy import stats


def success_runs_per_side(
    base_rate: float, drop: float, *, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Runs per side to detect a success-rate drop of `drop` from `base_rate`.

    Two-sided two-proportion test, normal approximation. It assumes independent
    (unpaired) batches, so it is conservative for paired comparisons, which
    need fewer runs when the two versions tend to fail the same seeds.
    """
    if not 0 < drop <= base_rate <= 1:
        raise ValueError("need 0 < drop <= base_rate <= 1")
    if not (0 < alpha < 1 and 0 < power < 1):
        raise ValueError("alpha and power must be between 0 and 1")
    p1, p2 = base_rate, base_rate - drop
    pooled = (p1 + p2) / 2
    z_alpha = float(stats.norm.ppf(1 - alpha / 2))
    z_power = float(stats.norm.ppf(power))
    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_power * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return math.ceil(numerator / drop**2)
