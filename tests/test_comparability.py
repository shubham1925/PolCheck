from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from polcheck.comparability import PAIRING_MIN_OVERLAP, check, detect_determinism
from polcheck.config import Config, RulesConfig
from polcheck.schema import UNKNOWN, Batch, BatchData, Run, SimulatorInfo

T0 = datetime(2026, 10, 5, tzinfo=UTC)
SIM = SimulatorInfo(name="mujoco", version="3.11.0")
SCENARIOS = {"near": "000000000000000a", "far": "000000000000000b", "side": "000000000000000c"}


def make_run(family: str, env_seed: int, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "family": family,
        "scenario_id": SCENARIOS[family],
        "env_seed": env_seed,
        "policy_seed": None,
        "simulator": SIM,
        "physics_hash": "ph1",
        "success": True,
        "metrics": {"score": 1.0},
    }
    fields.update(overrides)
    return fields


def make_batch(
    runs: Iterable[dict[str, Any]],
    *,
    suite_ref: str = "pick@1",
    deterministic: bool | None = True,
    policy_version: str = "v1",
) -> BatchData:
    batch_id = uuid4()
    built = [
        Run(
            run_id=uuid4(),
            batch_id=batch_id,
            policy_version=policy_version,
            suite_ref=suite_ref,
            source="native",
            has_timeseries=False,
            started_at=T0,
            **fields,
        )
        for fields in runs
    ]
    batch = Batch(
        batch_id=batch_id,
        policy_version=policy_version,
        suite_ref=suite_ref,
        created_at=T0,
        deterministic=deterministic,
        n_runs=len(built),
    )
    return BatchData(batch=batch, runs=built)


def runs_for(families: Mapping[str, Iterable[int]], **overrides: Any) -> list[dict[str, Any]]:
    return [make_run(f, s, **overrides) for f, seeds in families.items() for s in seeds]


STANDARD = {"near": range(30), "far": range(30)}


def config(**rules: str) -> Config:
    return Config(rules=RulesConfig(**rules))  # type: ignore[arg-type]


def rules_of(result: Any) -> dict[str, str]:
    return {i.rule: i.severity for i in result.issues}


# --- the happy path -------------------------------------------------------------


def test_identical_setup_is_comparable_and_paired() -> None:
    result = check(make_batch(runs_for(STANDARD)), make_batch(runs_for(STANDARD)), Config())
    assert result.status == "comparable"
    assert result.paired
    assert result.issues == []
    assert result.key_overlap == 1.0
    assert result.matched_keys == 60
    assert result.shared_scenarios == sorted([SCENARIOS["near"], SCENARIOS["far"]])
    assert [(f.family, f.n_base, f.n_cand) for f in result.families] == [
        ("far", 30, 30),
        ("near", 30, 30),
    ]
    assert result.low_n_families == []


# --- suite_ref --------------------------------------------------------------------


def test_different_suite_blocks() -> None:
    base = make_batch(runs_for(STANDARD), suite_ref="pick@1")
    cand = make_batch(runs_for(STANDARD), suite_ref="pick@2")
    result = check(base, cand, Config())
    assert result.status == "not_comparable"
    assert not result.paired
    assert rules_of(result) == {"suite_ref": "block"}
    assert "pick@1" in result.issues[0].message
    assert "pick@2" in result.issues[0].message


def test_suite_rule_can_be_relaxed() -> None:
    base = make_batch(runs_for(STANDARD), suite_ref="pick@1")
    cand = make_batch(runs_for(STANDARD), suite_ref="pick@2")
    result = check(base, cand, config(suite_ref="warn"))
    assert result.status == "comparable"
    assert result.paired
    assert rules_of(result) == {"suite_ref": "warn"}


# --- scenario_set -----------------------------------------------------------------


def test_partial_overlap_compares_intersection() -> None:
    base = make_batch(runs_for({**STANDARD, "side": range(30)}))
    cand = make_batch(runs_for({"near": range(30), "side": range(30)}))
    result = check(base, cand, Config())
    assert result.status == "partial"
    assert result.shared_scenarios == sorted([SCENARIOS["near"], SCENARIOS["side"]])
    assert [f.family for f in result.families] == ["near", "side"]
    assert result.paired  # keys are matched within the shared scenarios only
    (issue,) = result.issues
    assert (issue.rule, issue.severity) == ("scenario_set", "warn")
    assert "1 scenario(s) only in the baseline (families: far)" in issue.message
    assert "2 shared" in issue.message


def test_partial_overlap_can_block() -> None:
    base = make_batch(runs_for(STANDARD))
    cand = make_batch(runs_for({"near": range(30)}))
    assert check(base, cand, config(scenario_set="block")).status == "not_comparable"


def test_no_shared_scenarios_always_blocks() -> None:
    base = make_batch(runs_for({"near": range(30)}))
    cand = make_batch(runs_for({"far": range(30)}))
    result = check(base, cand, config(scenario_set="warn"))
    assert result.status == "not_comparable"
    assert rules_of(result)["scenario_set"] == "block"
    assert "no scenarios in common" in result.issues[0].message


# --- simulator_version and physics_hash ----------------------------------------


def test_simulator_version_warns() -> None:
    newer = SimulatorInfo(name="mujoco", version="3.11.1")
    base = make_batch(runs_for(STANDARD))
    cand = make_batch(runs_for(STANDARD, simulator=newer))
    result = check(base, cand, Config())
    assert result.status == "comparable"
    assert rules_of(result) == {"simulator_version": "warn"}
    assert "mujoco 3.11.0" in result.issues[0].message
    assert "mujoco 3.11.1" in result.issues[0].message
    assert check(base, cand, config(simulator_version="block")).status == "not_comparable"


def test_physics_mismatch_blocks() -> None:
    base = make_batch(runs_for(STANDARD))
    cand = make_batch(runs_for(STANDARD, physics_hash="ph2"))
    result = check(base, cand, Config())
    assert result.status == "not_comparable"
    assert not result.paired
    assert rules_of(result) == {"physics_hash": "block"}
    assert "ph1" in result.issues[0].message
    assert "Re-record" in result.issues[0].message
    relaxed = check(base, cand, config(physics_hash="warn"))
    assert relaxed.status == "comparable"


def test_mixed_physics_within_a_batch_is_reported() -> None:
    base = make_batch(runs_for(STANDARD))
    mixed = runs_for({"near": range(30)}) + runs_for({"far": range(30)}, physics_hash="ph2")
    result = check(base, make_batch(mixed), Config())
    assert result.status == "not_comparable"
    assert "ph1, ph2" in result.issues[0].message


@pytest.mark.parametrize("unknown_side", ["base", "cand", "both"])
def test_unknown_physics_warns_and_never_passes_silently(unknown_side: str) -> None:
    known, unknown = runs_for(STANDARD), runs_for(STANDARD, physics_hash=UNKNOWN)
    base = make_batch(unknown if unknown_side in ("base", "both") else known)
    cand = make_batch(unknown if unknown_side in ("cand", "both") else known)
    result = check(base, cand, config(physics_hash="block"))
    assert result.status == "comparable"
    assert rules_of(result) == {"physics_hash": "warn"}
    assert "cannot verify" in result.issues[0].message


def test_unknown_simulator_warns() -> None:
    unknown = SimulatorInfo(name=UNKNOWN, version=UNKNOWN)
    base = make_batch(runs_for(STANDARD, simulator=unknown))
    cand = make_batch(runs_for(STANDARD, simulator=unknown))
    result = check(base, cand, Config())
    assert rules_of(result) == {"simulator_version": "warn"}
    assert "baseline and candidate did not record it" in result.issues[0].message


# --- pairing ----------------------------------------------------------------------


def overlap_batches(base_seeds: range, cand_seeds: range) -> tuple[BatchData, BatchData]:
    return (
        make_batch(runs_for({"near": base_seeds})),
        make_batch(runs_for({"near": cand_seeds})),
    )


def test_pairing_at_exactly_ninety_percent_overlap() -> None:
    base, cand = overlap_batches(range(95), range(5, 100))  # 90 shared of 100 keys
    result = check(base, cand, Config())
    assert result.key_overlap == PAIRING_MIN_OVERLAP
    assert result.paired


def test_no_pairing_below_ninety_percent_overlap() -> None:
    base, cand = overlap_batches(range(95), range(6, 101))  # 89 shared of 101 keys
    result = check(base, cand, Config())
    assert result.key_overlap < PAIRING_MIN_OVERLAP
    assert not result.paired
    assert "only 88% of (scenario, env_seed) keys match" in result.pairing_reason


def test_unknown_determinism_needs_the_flag() -> None:
    base = make_batch(runs_for(STANDARD), deterministic=None)
    cand = make_batch(runs_for(STANDARD), deterministic=None)
    result = check(base, cand, Config())
    assert not result.paired
    assert "--assume-deterministic" in result.pairing_reason
    assumed = check(base, cand, Config(), assume_deterministic=True)
    assert assumed.paired
    assert "assumed deterministic" in assumed.pairing_reason


def test_known_nondeterminism_is_never_paired() -> None:
    base = make_batch(runs_for(STANDARD), deterministic=False)
    cand = make_batch(runs_for(STANDARD))
    result = check(base, cand, Config(), assume_deterministic=True)
    assert not result.paired
    assert "baseline is not deterministic" in result.pairing_reason


def test_missing_flag_falls_back_to_the_helper() -> None:
    repeated = [*runs_for(STANDARD), make_run("near", 0)]  # an identical repeat
    base = make_batch(repeated, deterministic=None)
    result = check(base, make_batch(runs_for(STANDARD)), Config())
    assert result.base_deterministic is True
    assert result.paired


# --- min_n -------------------------------------------------------------------------


def test_family_below_min_n_is_flagged() -> None:
    base = make_batch(runs_for({"near": range(19), "far": range(20)}))
    cand = make_batch(runs_for({"near": range(30), "far": range(20)}))
    result = check(base, cand, Config())
    assert result.status == "comparable"  # low n makes a family INCONCLUSIVE, not the batch
    assert result.low_n_families == ["near"]
    (issue,) = result.issues
    assert (issue.rule, issue.severity) == ("min_n", "warn")
    assert "Record 1 more baseline run(s)" in issue.message


def test_min_n_comes_from_config() -> None:
    base = make_batch(runs_for(STANDARD))
    result = check(base, make_batch(runs_for(STANDARD)), Config(min_n=31))
    assert result.low_n_families == ["far", "near"]


def test_deterministic_repeats_are_counted_once() -> None:
    repeats = [make_run("near", 0), make_run("near", 1)]
    det = make_batch(runs_for(STANDARD) + repeats, deterministic=True)
    nondet = make_batch(runs_for(STANDARD) + repeats, deterministic=False)
    other = make_batch(runs_for(STANDARD))
    near_det = check(det, other, Config()).families[1]
    near_nondet = check(nondet, other, Config()).families[1]
    assert (near_det.family, near_det.n_base) == ("near", 30)
    assert (near_nondet.family, near_nondet.n_base) == ("near", 32)


# --- determinism helper -----------------------------------------------------------


def built(runs: list[dict[str, Any]]) -> list[Run]:
    return make_batch(runs).runs


def test_no_repeats_is_unknown() -> None:
    assert detect_determinism(built(runs_for(STANDARD))) is None
    assert detect_determinism([]) is None


def test_identical_repeats_are_deterministic() -> None:
    assert detect_determinism(built([make_run("near", 3), make_run("near", 3)])) is True


@pytest.mark.parametrize(
    "change",
    [
        {"success": False},
        {"metrics": {"score": 1.0000001}},
        {"metrics": {"score": 1.0, "extra": 2.0}},
        {"failure_reason": "timeout"},
    ],
)
def test_differing_repeats_are_not_deterministic(change: dict[str, Any]) -> None:
    runs = built([make_run("near", 3), make_run("near", 3, **change), make_run("near", 4)])
    assert detect_determinism(runs) is False


def test_different_policy_seed_is_not_a_repeat() -> None:
    runs = built([make_run("near", 3, policy_seed=1), make_run("near", 3, policy_seed=2)])
    assert detect_determinism(runs) is None
