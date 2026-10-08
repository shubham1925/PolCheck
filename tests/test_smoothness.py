"""SPARC and LDLJ against values from the authors' reference implementation.

`fixtures/smoothness/reference.json` was produced by running siva82kb/SPARC
(scripts/smoothness.py, ISC licence) on the speed profiles stored in the file.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from polcheck.measures.sparc import dimensionless_jerk, sparc, uniform_speed
from tests.paths import FIXTURES

REFERENCE = json.loads((FIXTURES / "smoothness" / "reference.json").read_text())
CASES = REFERENCE["cases"]
TOLERANCE = 1e-3  # BUILD_PLAN M3 acceptance


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_sparc_matches_reference(case: dict[str, Any]) -> None:
    assert sparc(np.array(case["speed"]), case["fs"]) == pytest.approx(case["sparc"], abs=TOLERANCE)


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_dimensionless_jerk_matches_reference(case: dict[str, Any]) -> None:
    # The reference reports LDLJ = -ln|DJ|; ours reports ln(DJ) so higher is jerkier.
    ours = math.log(dimensionless_jerk(np.array(case["speed"]), case["fs"]))
    assert ours == pytest.approx(-case["ldlj_reference"], abs=TOLERANCE)


def test_reference_docstring_values_are_in_the_fixture() -> None:
    first = CASES[0]
    assert first["name"] == "docstring_gaussian"
    assert round(first["sparc"], 5) == -1.41403
    assert round(first["ldlj_reference"], 5) == -5.81636


def test_sub_movements_lower_smoothness() -> None:
    by_name = {c["name"]: c for c in CASES}
    close, apart = by_name["gaussian_pair_sep0.2"], by_name["gaussian_pair_sep0.8"]
    assert sparc(np.array(apart["speed"]), 100.0) < sparc(np.array(close["speed"]), 100.0) - 1.0


def test_all_zero_speed_is_rejected() -> None:
    with pytest.raises(ValueError, match="all zero"):
        sparc(np.zeros(50), 25.0)
    with pytest.raises(ValueError, match="all zero"):
        dimensionless_jerk(np.zeros(50), 25.0)


def test_uniform_speed_from_positions() -> None:
    t = np.arange(0, 1.0001, 0.1)
    pos = np.column_stack([0.3 * t, 0.4 * t, np.zeros_like(t)])  # 0.5 m/s straight line
    speed, fs = uniform_speed(t, pos)
    assert fs == pytest.approx(10.0)
    np.testing.assert_allclose(speed, 0.5)


def test_uniform_speed_resamples_irregular_timestamps() -> None:
    rng = np.random.default_rng(0)
    t_regular = np.arange(0, 2.0, 0.04)
    t_jittered = t_regular + rng.uniform(-0.008, 0.008, t_regular.size)
    t_jittered[0], t_jittered[-1] = t_regular[0], t_regular[-1]

    def path(t: np.ndarray) -> np.ndarray:
        s = t / 2.0
        return np.column_stack(
            [10 * s**3 - 15 * s**4 + 6 * s**5, np.zeros_like(t), np.zeros_like(t)]
        )

    regular = sparc(*uniform_speed(t_regular, path(t_regular)))
    jittered = sparc(*uniform_speed(t_jittered, path(t_jittered)))
    assert jittered == pytest.approx(regular, abs=0.05)
