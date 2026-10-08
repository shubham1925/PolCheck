"""Smoothness metrics on a speed profile (Balasubramanian et al. 2015).

`sparc` and `dimensionless_jerk` follow the authors' reference implementation
(github.com/siva82kb/SPARC, scripts/smoothness.py); tests check them against
values produced by that code (tests/fixtures/smoothness/reference.json).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

SPARC_PADLEVEL = 4
SPARC_FC = 10.0  # Hz, maximum cut-off frequency
SPARC_AMP_TH = 0.05  # amplitude threshold for the adaptive cut-off


def sparc(
    speed: FloatArray,
    fs: float,
    padlevel: int = SPARC_PADLEVEL,
    fc: float = SPARC_FC,
    amp_th: float = SPARC_AMP_TH,
) -> float:
    """Spectral arc length of a speed profile. More negative is less smooth.

    Raises ValueError for a profile that never moves (its spectrum is all zero).
    """
    speed = np.asarray(speed, dtype=np.float64)
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(nfft) * (fs / nfft)
    magnitude = np.abs(np.fft.fft(speed, nfft))
    peak = magnitude.max()
    if peak == 0:
        raise ValueError("speed profile is all zero")
    magnitude = magnitude / peak

    # Low-pass to fc, then keep the span where the spectrum is above amp_th.
    within = f <= fc
    f_sel, m_sel = f[within], magnitude[within]
    above = np.nonzero(m_sel >= amp_th)[0]
    f_sel = f_sel[above[0] : above[-1] + 1]
    m_sel = m_sel[above[0] : above[-1] + 1]
    if len(f_sel) < 2:
        return 0.0  # a single spectral point has no arc length
    df = np.diff(f_sel) / (f_sel[-1] - f_sel[0])
    return float(-np.sum(np.sqrt(df**2 + np.diff(m_sel) ** 2)))


def dimensionless_jerk(speed: FloatArray, fs: float) -> float:
    """Dimensionless jerk of a speed profile, as a positive number (the reference
    returns it negated). Larger is jerkier."""
    speed = np.asarray(speed, dtype=np.float64)
    peak = np.abs(speed).max()
    if peak == 0:
        raise ValueError("speed profile is all zero")
    dt = 1.0 / fs
    duration = len(speed) * dt
    jerk = np.diff(speed, 2) / dt**2
    return float(duration**3 / peak**2 * np.sum(jerk**2) * dt)


def uniform_speed(t: FloatArray, pos: FloatArray) -> tuple[FloatArray, float]:
    """Speed profile and sampling rate from positions `pos` (n x d) at times `t`.

    Spectral measures need uniform sampling, so non-uniform timestamps are first
    resampled (linear interpolation) onto a grid at the median time step.
    """
    t = np.asarray(t, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64).reshape(len(t), -1)
    steps = np.diff(t)
    dt = float(np.median(steps))
    if not np.allclose(steps, dt, rtol=1e-6, atol=1e-12):
        grid = t[0] + dt * np.arange(int(np.floor((t[-1] - t[0]) / dt)) + 1)
        pos = np.column_stack([np.interp(grid, t, pos[:, k]) for k in range(pos.shape[1])])
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
    return speed, 1.0 / dt
