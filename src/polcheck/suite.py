"""Suite loading and scenario fingerprinting (BUILD_PLAN 6.1)."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

from pydantic import JsonValue, ValidationError

from polcheck.schema import Scenario, Suite


class SuiteError(ValueError):
    pass


def canonical_json(value: JsonValue) -> str:
    """Sorted keys, no whitespace, floats as repr, ASCII only.

    Changing this changes every scenario id, so it is effectively frozen.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def scenario_id(scenario: Scenario) -> str:
    payload = canonical_json({"family": scenario.family, "config": scenario.config})
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


def suite_hash(suite: Suite) -> str:
    """Hash of a suite's content, ignoring TOML formatting and comments."""
    payload = canonical_json(suite.model_dump(mode="json"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def family_scenarios(suite: Suite) -> dict[str, list[str]]:
    """Scenario ids per family, in file order (the order defines `scenario_index`)."""
    out: dict[str, list[str]] = {}
    for sc in suite.scenarios:
        out.setdefault(sc.family, []).append(scenario_id(sc))
    return out


def parse_suite(text: str, *, source: str = "<string>") -> Suite:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise SuiteError(f"{source}: invalid TOML: {exc}") from exc
    unknown = sorted(set(data) - {"suite", "scenario"})
    if unknown:
        raise SuiteError(f"{source}: unknown top-level keys: {', '.join(unknown)}")
    header = data.get("suite")
    if not isinstance(header, dict):
        raise SuiteError(f"{source}: missing [suite] table with name and version")
    if "scenarios" in header:
        raise SuiteError(f"{source}: define scenarios as [[scenario]] tables, not in [suite]")
    try:
        return Suite.model_validate({**header, "scenarios": data.get("scenario", [])})
    except ValidationError as exc:
        raise SuiteError(f"{source}: {exc}") from exc


def load_suite(path: Path) -> Suite:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SuiteError(f"suite file not found: {path}") from exc
    return parse_suite(text, source=str(path))
