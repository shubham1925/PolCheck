"""Bring time-series logged at different rates onto a common rate (DECISIONS.md D31).

Both batches go through the same steps, including the one already at the common
rate, so they end up with the same frequency content:

1. interpolate onto a uniform grid at the run's own rate (removes jitter),
2. low-pass at 80% of the common rate's Nyquist frequency (zero-phase),
3. interpolate onto a uniform grid at the common rate.

Filtering only the faster side would compare a clean signal against one that
still carries whatever aliasing the slower recording has.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from scipy import signal

CUTOFF_FRACTION = 0.8
"""Low-pass cutoff as a share of the common rate's Nyquist frequency."""

FILTER_ORDER = 4

TOO_SHORT = "too short to low-pass; interpolated only"


def _uniform_grid(t0: float, t1: float, hz: float) -> np.ndarray:
    n = int(np.floor((t1 - t0) * hz + 1e-9)) + 1
    return np.asarray(t0 + np.arange(n) / hz)


def resample_table(table: pa.Table, target_hz: float) -> tuple[pa.Table, str | None]:
    """Resample every signal in a time-series table to `target_hz`.

    Returns the new table and a note if the run was too short to filter. A run
    with fewer than two samples is returned unchanged.
    """
    t = np.asarray(table.column("t").to_numpy(), dtype=np.float64)
    if len(t) < 2:
        return table, TOO_SHORT
    own_hz = 1.0 / float(np.median(np.diff(t)))
    own_grid = _uniform_grid(t[0], t[-1], own_hz)
    target_grid = _uniform_grid(t[0], t[-1], target_hz)

    cutoff = CUTOFF_FRACTION * target_hz / 2
    sos = None
    if cutoff < own_hz / 2:  # always true unless this run is slower than the target
        sos = signal.butter(FILTER_ORDER, cutoff, fs=own_hz, output="sos")
    # sosfiltfilt pads each end; it needs more samples than its default padding.
    pad = 3 * (2 * len(sos) + 1) if sos is not None else 0
    can_filter = sos is not None and len(own_grid) > pad

    columns: dict[str, pa.Array] = {"t": pa.array(target_grid, pa.float64())}
    for name in table.column_names:
        if name == "t":
            continue
        values = np.interp(own_grid, t, np.asarray(table.column(name).to_numpy(), np.float64))
        if can_filter:
            values = signal.sosfiltfilt(sos, values)
        columns[name] = pa.array(np.interp(target_grid, own_grid, values), pa.float64())
    return pa.table(columns), None if can_filter else TOO_SHORT
