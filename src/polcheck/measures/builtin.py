"""The built-in per-run measures (BUILD_PLAN 7.3).

Only for setups whose evaluation tool does not already provide them. Default
thresholds are starting points; see DECISIONS.md D27 for the reasoning, and
override them per measure in polcheck.toml.
"""

from __future__ import annotations

import numpy as np

from polcheck.measures import CONTACTS, RunData, measure
from polcheck.measures.sparc import dimensionless_jerk, sparc, uniform_speed

HESITATION_SPEED = 0.01  # m/s: slower than this counts as hesitating
HESITATION_TRIM = 0.1  # ignore the first and last 10% of the run


def _speed(run: RunData) -> tuple[np.ndarray, float] | None:
    t, pos = run.ts.t, run.ts["ee_pos"]
    if len(t) < 3:
        return None
    return uniform_speed(t, pos)


@measure(
    name="sparc",
    worse="lower",
    threshold=0.1,
    requires=["t", "ee_pos"],
)
def sparc_measure(run: RunData) -> float | None:
    """Spectral arc length of end-effector speed; more negative is less smooth."""
    profile = _speed(run)
    if profile is None or not profile[0].any():
        return None
    return sparc(*profile)


@measure(
    name="log_dimensionless_jerk",
    worse="higher",
    threshold=0.5,
    requires=["t", "ee_pos"],
)
def log_dimensionless_jerk(run: RunData) -> float | None:
    """ln of the dimensionless jerk of end-effector speed; higher is jerkier."""
    profile = _speed(run)
    if profile is None or not profile[0].any():
        return None
    dj = dimensionless_jerk(*profile)
    return float(np.log(dj)) if dj > 0 else None


@measure(
    name="peak_unintended_contact_force",
    worse="higher",
    threshold=5.0,
    unit="N",
    requires=[CONTACTS],
)
def peak_unintended_contact_force(run: RunData) -> float | None:
    """Largest contact force among contacts not marked intended (0 if none)."""
    contacts = run.contacts
    forces = contacts.column("force_norm").to_numpy()
    unintended = ~contacts.column("intended").to_numpy(zero_copy_only=False).astype(bool)
    return float(forces[unintended].max()) if unintended.any() else 0.0


@measure(
    name="min_clearance",
    worse="lower",
    threshold=0.01,
    unit="m",
    requires=["clearance"],
)
def min_clearance(run: RunData) -> float | None:
    """Closest the robot came to its surroundings."""
    clearance = run.ts["clearance"]
    return float(np.min(clearance)) if len(clearance) else None


@measure(
    name="hesitation_time",
    worse="higher",
    threshold=0.2,
    unit="s",
    requires=["t", "ee_pos"],
)
def hesitation_time(run: RunData) -> float | None:
    """Seconds the end-effector moved slower than 1 cm/s, ignoring the first and
    last 10% of the run."""
    t, pos = run.ts.t, run.ts["ee_pos"]
    if len(t) < 2:
        return None
    duration = t[-1] - t[0]
    if duration <= 0:
        return None
    dt = np.diff(t)
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
    # Count each interval by the part of it inside the trimmed window.
    lo, hi = t[0] + HESITATION_TRIM * duration, t[-1] - HESITATION_TRIM * duration
    inside = np.clip(np.minimum(t[1:], hi) - np.maximum(t[:-1], lo), 0.0, None)
    return float(inside[speed < HESITATION_SPEED].sum())


@measure(
    name="task_time",
    worse="higher",
    threshold=0.25,
    unit="s",
    requires=["t"],
    success_only=True,
)
def task_time(run: RunData) -> float | None:
    """Duration of a successful run."""
    t = run.ts.t
    return float(t[-1] - t[0]) if len(t) else None
