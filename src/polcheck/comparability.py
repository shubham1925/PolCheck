"""Comparability checker and determinism helper (BUILD_PLAN 7.4).

Decides whether two batches can be compared fairly, over which scenarios, and
whether the comparison can pair runs seed by seed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

from polcheck.config import Config
from polcheck.schema import (
    UNKNOWN,
    BatchData,
    ComparabilityResult,
    ComparabilityStatus,
    FamilyCounts,
    Issue,
    Run,
    Severity,
)

PAIRING_MIN_OVERLAP = 0.9
"""Share of `(scenario_id, env_seed)` keys that must match for a paired comparison."""

RunKey = tuple[str, int]


# --- Determinism helper -----------------------------------------------------


def _outcome(run: Run) -> tuple[object, ...]:
    return (run.success, run.failure_reason, sorted(run.metrics.items()))


def detect_determinism(runs: Sequence[Run]) -> bool | None:
    """Whether repeated `(scenario_id, env_seed, policy_seed)` keys give identical
    summaries (success, failure reason, metrics, compared exactly).

    True if at least one key repeats and every repeat matches; False if any
    repeat differs; None if no key repeats, so there is no evidence either way.
    """
    seen: dict[tuple[str, int, int | None], tuple[object, ...]] = {}
    repeated = False
    for run in runs:
        key = (run.scenario_id, run.env_seed, run.policy_seed)
        outcome = _outcome(run)
        if key in seen:
            if seen[key] != outcome:
                return False
            repeated = True
        else:
            seen[key] = outcome
    return True if repeated else None


def _determinism(data: BatchData) -> bool | None:
    """The batch's recorded flag, or the helper's verdict for older batches."""
    if data.batch.deterministic is not None:
        return data.batch.deterministic
    return detect_determinism(data.runs)


# --- Rules ------------------------------------------------------------------


def _fmt(values: set[str]) -> str:
    return ", ".join(sorted(values))


def _check_suite_ref(base: BatchData, cand: BatchData, config: Config) -> list[Issue]:
    b, c = base.batch.suite_ref, cand.batch.suite_ref
    if b == c:
        return []
    return [
        Issue(
            rule="suite_ref",
            severity=config.rules.suite_ref,
            message=(
                f"baseline was run on suite {b}, candidate on {c}. Re-run the candidate "
                f"on {b} (or record a new baseline on {c}) so both use the same scenarios."
            ),
        )
    ]


def _family_names(runs: Sequence[Run], ids: set[str]) -> str:
    families = sorted({r.family for r in runs if r.scenario_id in ids})
    return ", ".join(families)


def _check_scenario_set(
    base: BatchData, cand: BatchData, config: Config, shared: set[str]
) -> list[Issue]:
    base_ids = {r.scenario_id for r in base.runs}
    cand_ids = {r.scenario_id for r in cand.runs}
    if base_ids == cand_ids:
        return []
    if not shared:
        return [
            Issue(
                rule="scenario_set",
                severity="block",  # nothing to compare, whatever the configured severity
                message=(
                    "the batches have no scenarios in common, so nothing can be compared. "
                    "Run both versions on the same suite."
                ),
            )
        ]
    parts = []
    if only_base := base_ids - cand_ids:
        parts.append(
            f"{len(only_base)} scenario(s) only in the baseline "
            f"(families: {_family_names(base.runs, only_base)})"
        )
    if only_cand := cand_ids - base_ids:
        parts.append(
            f"{len(only_cand)} scenario(s) only in the candidate "
            f"(families: {_family_names(cand.runs, only_cand)})"
        )
    return [
        Issue(
            rule="scenario_set",
            severity=config.rules.scenario_set,
            message=(
                f"{'; '.join(parts)}. Comparing only the {len(shared)} shared scenario(s); "
                "run the missing scenarios on both versions for a full comparison."
            ),
        )
    ]


def _check_setting(
    rule: str,
    what: str,
    base_values: set[str],
    cand_values: set[str],
    unknown_sides: list[str],
    severity: Severity,
    fix: str,
) -> list[Issue]:
    """Shared logic for simulator version and physics hash.

    A side that did not record the setting cannot be verified: that is always a
    warning, never a pass (two unknowns are not evidence of a match) and never a
    block.
    """
    if unknown_sides:
        return [
            Issue(
                rule=rule,
                severity="warn",
                message=(
                    f"cannot verify that both batches used the same {what}: the "
                    f"{' and '.join(unknown_sides)} did not record it. "
                    "Set it when ingesting (or in [readers.tabular]) to enable this check."
                ),
            )
        ]
    if base_values == cand_values:
        return []
    return [
        Issue(
            rule=rule,
            severity=severity,
            message=(
                f"baseline {what}: {_fmt(base_values)}; candidate {what}: "
                f"{_fmt(cand_values)}. {fix}"
            ),
        )
    ]


def _unknown_sides(
    base: BatchData, cand: BatchData, is_unknown: Callable[[Run], bool]
) -> list[str]:
    return [
        side
        for side, data in (("baseline", base), ("candidate", cand))
        if any(is_unknown(r) for r in data.runs)
    ]


def _check_simulator(base: BatchData, cand: BatchData, config: Config) -> list[Issue]:
    def versions(data: BatchData) -> set[str]:
        return {f"{r.simulator.name} {r.simulator.version}" for r in data.runs}

    return _check_setting(
        "simulator_version",
        "simulator",
        versions(base),
        versions(cand),
        _unknown_sides(base, cand, lambda r: UNKNOWN in (r.simulator.name, r.simulator.version)),
        config.rules.simulator_version,
        "Results can shift between simulator versions; re-record one side to match.",
    )


def _check_physics(base: BatchData, cand: BatchData, config: Config) -> list[Issue]:
    return _check_setting(
        "physics_hash",
        "physics settings (hash)",
        {r.physics_hash for r in base.runs},
        {r.physics_hash for r in cand.runs},
        _unknown_sides(base, cand, lambda r: r.physics_hash == UNKNOWN),
        config.rules.physics_hash,
        "Timestep, solver or other physics settings differ, so the runs are not "
        "comparable. Re-record both versions with the same simulator settings.",
    )


# --- Pairing and sample sizes ----------------------------------------------


def _keys(runs: Sequence[Run], shared: set[str]) -> set[RunKey]:
    return {(r.scenario_id, r.env_seed) for r in runs if r.scenario_id in shared}


def _describe(flag: bool | None) -> str:
    return {True: "deterministic", False: "not deterministic", None: "of unknown determinism"}[flag]


def _pairing(
    base_det: bool | None,
    cand_det: bool | None,
    overlap: float,
    assume_deterministic: bool,
) -> tuple[bool, str]:
    def usable(flag: bool | None) -> bool:
        return flag is True or (flag is None and assume_deterministic)

    if not (usable(base_det) and usable(cand_det)):
        reason = (
            f"unpaired: the baseline is {_describe(base_det)} and the candidate is "
            f"{_describe(cand_det)}. Paired tests need both to be deterministic"
        )
        if base_det is None or cand_det is None:
            reason += (
                " (record a few repeated seeds so it can be checked, or pass "
                "--assume-deterministic if you know it is)"
            )
        return False, reason + "."
    if overlap < PAIRING_MIN_OVERLAP:
        return False, (
            f"unpaired: only {overlap:.0%} of (scenario, env_seed) keys match, below the "
            f"{PAIRING_MIN_OVERLAP:.0%} needed. Run both versions on the same seeds to pair them."
        )
    assumed = " (assumed deterministic)" if None in (base_det, cand_det) else ""
    return True, f"paired on {overlap:.0%} matching (scenario, env_seed) keys{assumed}."


def _family_counts(
    base: BatchData, cand: BatchData, shared: set[str], base_det: bool | None, cand_det: bool | None
) -> list[FamilyCounts]:
    """Runs per family over shared scenarios. In a deterministic batch a repeated
    key is an exact copy and adds no information, so it is counted once."""

    def count(runs: Sequence[Run], det: bool | None) -> Counter[str]:
        counts: Counter[str] = Counter()
        seen: set[tuple[str, int, int | None]] = set()
        for r in runs:
            if r.scenario_id not in shared:
                continue
            key = (r.scenario_id, r.env_seed, r.policy_seed)
            if det and key in seen:
                continue
            seen.add(key)
            counts[r.family] += 1
        return counts

    b, c = count(base.runs, base_det), count(cand.runs, cand_det)
    return [FamilyCounts(family=f, n_base=b[f], n_cand=c[f]) for f in sorted(b.keys() | c.keys())]


def _check_min_n(families: list[FamilyCounts], min_n: int) -> tuple[list[str], list[Issue]]:
    low = [f for f in families if f.n_base < min_n or f.n_cand < min_n]
    issues = []
    for f in low:
        short = [
            f"{min_n - n} more {side} run(s)"
            for side, n in (("baseline", f.n_base), ("candidate", f.n_cand))
            if n < min_n
        ]
        issues.append(
            Issue(
                rule="min_n",
                severity="warn",
                message=(
                    f"family {f.family} has {f.n_base} baseline and {f.n_cand} candidate "
                    f"run(s), below min_n = {min_n}, so it will be reported INCONCLUSIVE. "
                    f"Record {' and '.join(short)}."
                ),
            )
        )
    return [f.family for f in low], issues


# --- Entry point ------------------------------------------------------------


def check(
    base: BatchData,
    cand: BatchData,
    config: Config,
    *,
    assume_deterministic: bool = False,
) -> ComparabilityResult:
    base_ids = {r.scenario_id for r in base.runs}
    cand_ids = {r.scenario_id for r in cand.runs}
    shared = base_ids & cand_ids

    issues = [
        *_check_suite_ref(base, cand, config),
        *_check_scenario_set(base, cand, config, shared),
        *_check_simulator(base, cand, config),
        *_check_physics(base, cand, config),
    ]

    base_det, cand_det = _determinism(base), _determinism(cand)
    base_keys, cand_keys = _keys(base.runs, shared), _keys(cand.runs, shared)
    union = base_keys | cand_keys
    matched = len(base_keys & cand_keys)
    overlap = matched / len(union) if union else 0.0
    paired, reason = _pairing(base_det, cand_det, overlap, assume_deterministic)

    families = _family_counts(base, cand, shared, base_det, cand_det)
    low_n, min_n_issues = _check_min_n(families, config.min_n)
    issues += min_n_issues

    status: ComparabilityStatus
    if any(i.severity == "block" for i in issues):
        status = "not_comparable"
    elif base_ids != cand_ids:
        status = "partial"
    else:
        status = "comparable"

    if status == "not_comparable":
        paired, reason = False, "not compared: see the blocking issues."

    return ComparabilityResult(
        status=status,
        paired=paired,
        pairing_reason=reason,
        matched_keys=matched,
        key_overlap=overlap,
        base_deterministic=base_det,
        cand_deterministic=cand_det,
        shared_scenarios=sorted(shared),
        families=families,
        low_n_families=low_n,
        issues=issues,
    )
