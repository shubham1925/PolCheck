from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from polcheck.schema import Scenario, SeedRange
from polcheck.suite import (
    SuiteError,
    canonical_json,
    family_scenarios,
    load_suite,
    parse_suite,
    scenario_id,
    suite_hash,
)
from tests.paths import PICK_SUITE, SHELF_SUITE

SEEDS = SeedRange(start=0, count=1)

# Pinned so an accidental change to the fingerprint (which would orphan every
# stored batch) fails loudly.
NEAR_LEFT_ID = scenario_id(
    Scenario(
        family="near-left", config={"object_x": [-0.15, 0.0], "object_y": [0.0, 0.15]}, seeds=SEEDS
    )
)


def test_canonical_json_format() -> None:
    assert canonical_json({"b": [1, 0.1, 1e16], "a": {"z": None, "y": True}}) == (
        '{"a":{"y":true,"z":null},"b":[1,0.1,1e+16]}'
    )


def test_scenario_id_golden_value() -> None:
    assert scenario_id(Scenario(family="f", config={"x": 0.5}, seeds=SEEDS)) == ("63855579b19784c1")


def test_scenario_id_ignores_key_order_and_seeds() -> None:
    a = Scenario(family="f", config={"x": 1.0, "y": {"p": 1, "q": 2}}, seeds=SEEDS)
    b = Scenario(
        family="f", config={"y": {"q": 2, "p": 1}, "x": 1.0}, seeds=SeedRange(start=5, count=9)
    )
    assert scenario_id(a) == scenario_id(b)


def test_scenario_id_distinguishes_family_config_and_number_type() -> None:
    base = scenario_id(Scenario(family="f", config={"x": 1.0}, seeds=SEEDS))
    assert base != scenario_id(Scenario(family="g", config={"x": 1.0}, seeds=SEEDS))
    assert base != scenario_id(Scenario(family="f", config={"x": 1.5}, seeds=SEEDS))
    assert base != scenario_id(Scenario(family="f", config={"x": 1}, seeds=SEEDS))


@pytest.mark.parametrize("hash_seed", ["0", "1", "12345"])
def test_scenario_id_stable_across_processes(hash_seed: str) -> None:
    code = (
        "from pathlib import Path; from polcheck.suite import load_suite, scenario_id; "
        f"print(scenario_id(load_suite(Path({str(PICK_SUITE)!r})).scenarios[0]))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
    )
    assert out.stdout.strip() == NEAR_LEFT_ID


def test_load_suite() -> None:
    suite = load_suite(PICK_SUITE)
    assert suite.ref == "pick@1"
    assert [sc.family for sc in suite.scenarios] == [
        "near-left",
        "near-right",
        "far-left",
        "far-right",
    ]
    assert scenario_id(suite.scenarios[0]) == NEAR_LEFT_ID


def test_family_scenarios_keeps_file_order() -> None:
    suite = load_suite(SHELF_SUITE)
    groups = family_scenarios(suite)
    assert list(groups) == ["shelf", "floor"]
    assert groups["shelf"] == [scenario_id(suite.scenarios[0]), scenario_id(suite.scenarios[1])]


def test_suite_hash_ignores_formatting() -> None:
    compact = '[suite]\nname="s"\nversion=1\n[[scenario]]\nfamily="f"\nseeds={start=0,count=1}\n'
    spaced = (
        "# comment\n[suite]\nversion = 1\nname = 's'\n\n"
        "[[scenario]]\nseeds = { count = 1, start = 0 }\nfamily = 'f'\n"
    )
    assert suite_hash(parse_suite(compact)) == suite_hash(parse_suite(spaced))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not toml [", "invalid TOML"),
        ('[[scenario]]\nfamily="f"\nseeds={start=0,count=1}\n', "missing [suite]"),
        ('[suite]\nname="s"\nversion=1\n', "scenarios"),
        (
            '[suite]\nname="s@x"\nversion=1\n[[scenario]]\nfamily="f"\nseeds={start=0,count=1}\n',
            "name",
        ),
        (
            '[suite]\nname="s"\nversion=0\n[[scenario]]\nfamily="f"\nseeds={start=0,count=1}\n',
            "version",
        ),
        ('[suite]\nname="s"\nversion=1\n[extra]\n', "unknown top-level keys: extra"),
        (
            '[suite]\nname="s"\nversion=1\n'
            '[[scenario]]\nfamily="f"\nconfig={a=1}\nseeds={start=0,count=1}\n'
            '[[scenario]]\nfamily="f"\nconfig={a=1}\nseeds={start=5,count=1}\n',
            "same config twice",
        ),
    ],
)
def test_invalid_suites_rejected(text: str, message: str) -> None:
    with pytest.raises(SuiteError, match=re.escape(message)):
        parse_suite(text)


def test_missing_suite_file(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="not found"):
        load_suite(tmp_path / "nope.toml")
