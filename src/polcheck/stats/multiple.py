"""Multiple-comparison correction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, in the input order.

    Rejecting every test whose adjusted p-value is below q keeps the expected
    share of false discoveries among the rejections at or below q.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)
    # Step-up: each adjusted value is the smallest scaled value at or after its rank.
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1].clip(max=1.0)
    out = np.empty(m)
    out[order] = adjusted
    return [float(v) for v in out]
