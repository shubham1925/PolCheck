"""Turn test results into verdicts (BUILD_PLAN 7.5)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from polcheck.measures import MeasureSpec
from polcheck.schema import Role, Verdict
from polcheck.stats import RawTest

_EPS = 1e-12  # so an estimate that equals the threshold up to rounding still counts


def threshold_for(spec: MeasureSpec) -> float:
    """The smallest change that matters for a measure, in its own units.

    The only place a test's threshold is decided. OPEN_QUESTIONS 11 (absolute
    vs relative thresholds) is settled by changing this function alone.
    """
    if spec.threshold is None:
        raise ValueError(f"measure {spec.name!r} has no threshold")
    return spec.threshold


def judge(
    raw: RawTest | None, p_adjusted: float | None, threshold: float, q: float, *, enough_runs: bool
) -> Verdict:
    """REGRESSED needs all three: statistically real after correction, the
    interval excludes zero, and the estimated change is at least the threshold.
    PASS needs the whole interval below the threshold. Anything else, including
    too few runs to test, is INCONCLUSIVE."""
    if raw is None or p_adjusted is None or not enough_runs:
        return "INCONCLUSIVE"
    if p_adjusted < q and raw.ci_low > 0 and raw.estimate >= threshold - _EPS:
        return "REGRESSED"
    if raw.ci_high < threshold:
        return "PASS"
    return "INCONCLUSIVE"


class _Judged(Protocol):
    @property
    def verdict(self) -> Verdict: ...
    @property
    def role(self) -> Role: ...
    @property
    def info_only(self) -> bool: ...


def overall_verdict(tests: Iterable[_Judged]) -> Verdict:
    """Gate/advisory rollup. Info-only tests never count.

    REGRESSED if any test regressed (gate or advisory); otherwise INCONCLUSIVE
    if any gate test is inconclusive; otherwise PASS.
    """
    counted = [t for t in tests if not t.info_only]
    if any(t.verdict == "REGRESSED" for t in counted):
        return "REGRESSED"
    if any(t.verdict == "INCONCLUSIVE" and t.role == "gate" for t in counted):
        return "INCONCLUSIVE"
    return "PASS"
