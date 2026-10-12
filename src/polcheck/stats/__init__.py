"""Statistical tests behind a comparison (BUILD_PLAN 7.5).

Everything here works on plain numbers and arrays, with no simulator, store or
measure registry, so it can be tested and validated in isolation. All estimates
are oriented so that **positive means the candidate is worse**.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RawTest:
    """One test's result before multiple-comparison correction and verdict."""

    estimate: float
    """Oriented change, positive = worse."""
    ci_low: float
    ci_high: float
    p_value: float
    method: str


def rng_for(seed: int, *labels: str) -> np.random.Generator:
    """An independent, reproducible generator for one test.

    Keyed by the test's labels (family, measure, ...), not by the order tests run
    in, so adding or removing a test never changes another test's result.
    """
    key = zlib.crc32("|".join(labels).encode())
    return np.random.default_rng([seed, key])
