"""Generic per-episode CSV / JSONL reader (BUILD_PLAN 7.2).

One row per episode. Column names come from `[readers.tabular]` in the config.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.json as pa_json

from polcheck.comparability import detect_determinism
from polcheck.config import Config
from polcheck.readers.base import ReaderError
from polcheck.schema import UNKNOWN, Batch, BatchData, Run, SimulatorInfo, Suite
from polcheck.suite import family_scenarios

_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


def _load(path: Path) -> pa.Table:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            # Empty cells are null, not "", so an empty failure_reason means none.
            options = pa_csv.ConvertOptions(strings_can_be_null=True)
            return pa_csv.read_csv(path, convert_options=options)
        if suffix in (".jsonl", ".ndjson"):
            return pa_json.read_json(path)
    except FileNotFoundError as exc:
        raise ReaderError(f"file not found: {path}") from exc
    except pa.ArrowInvalid as exc:
        raise ReaderError(f"{path}: could not parse: {exc}") from exc
    raise ReaderError(f"{path}: unsupported file type {suffix!r} (use .csv, .jsonl or .ndjson)")


def _is_numeric(type_: pa.DataType) -> bool:
    return bool(pa.types.is_integer(type_) or pa.types.is_floating(type_))


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool):
        raise ReaderError(f"{where}: expected an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise ReaderError(f"{where}: expected an integer, got {value!r}")


def _as_bool(value: Any, where: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
    raise ReaderError(f"{where}: expected true/false or 1/0, got {value!r}")


class TabularReader:
    name = "tabular"

    def read(self, path: Path, *, policy_version: str, suite: Suite, config: Config) -> BatchData:
        mapping = config.readers.tabular
        table = _load(path)
        if table.num_rows == 0:
            raise ReaderError(f"{path}: no rows")
        columns = set(table.column_names)

        mapped = [mapping.family, mapping.env_seed, mapping.success]
        mapped += [
            c
            for c in (mapping.policy_seed, mapping.failure_reason, mapping.scenario_index)
            if c is not None
        ]
        missing = [c for c in [*mapped, *(mapping.metrics or [])] if c not in columns]
        if missing:
            raise ReaderError(
                f"{path}: missing columns {', '.join(missing)} "
                "(column names come from [readers.tabular] in polcheck.toml)"
            )
        if mapping.metrics is not None:
            metric_columns = list(mapping.metrics)
        else:
            metric_columns = [
                field.name
                for field in table.schema
                if field.name not in mapped and _is_numeric(field.type)
            ]

        rows = table.to_pylist()
        scenarios = family_scenarios(suite)
        families = {str(r[mapping.family]) for r in rows if r[mapping.family] is not None}
        unknown = sorted(families - scenarios.keys())
        if unknown:
            raise ReaderError(
                f"{path}: families not in suite {suite.ref}: {', '.join(unknown)} "
                f"(suite families: {', '.join(scenarios)})"
            )
        multi = sorted(f for f in families if len(scenarios[f]) > 1)
        if multi and mapping.scenario_index is None:
            raise ReaderError(
                f"{path}: families {', '.join(multi)} have several scenarios in suite "
                f"{suite.ref}; set scenario_index in [readers.tabular] to the column that "
                "gives each row's scenario (0-based, in suite file order)"
            )

        batch_id = uuid4()
        simulator = mapping.simulator or SimulatorInfo(name=UNKNOWN, version=UNKNOWN)
        physics_hash = mapping.physics_hash or UNKNOWN
        now = datetime.now(UTC)
        runs = []
        for i, row in enumerate(rows, start=1):
            where = f"{path}: row {i}"
            family_value = row[mapping.family]
            if family_value is None:
                raise ReaderError(f"{where}: {mapping.family} is empty")
            family = str(family_value)
            ids = scenarios[family]
            if len(ids) == 1:
                sid = ids[0]
            else:
                assert mapping.scenario_index is not None
                index = _as_int(row[mapping.scenario_index], f"{where}, {mapping.scenario_index}")
                if not 0 <= index < len(ids):
                    raise ReaderError(
                        f"{where}: scenario index {index} out of range for family "
                        f"{family!r} ({len(ids)} scenarios)"
                    )
                sid = ids[index]
            policy_seed = None
            if mapping.policy_seed is not None and row[mapping.policy_seed] is not None:
                policy_seed = _as_int(row[mapping.policy_seed], f"{where}, {mapping.policy_seed}")
            failure_reason = None
            if mapping.failure_reason is not None and row[mapping.failure_reason] is not None:
                failure_reason = str(row[mapping.failure_reason])
            runs.append(
                Run(
                    run_id=uuid4(),
                    batch_id=batch_id,
                    policy_version=policy_version,
                    suite_ref=suite.ref,
                    scenario_id=sid,
                    family=family,
                    env_seed=_as_int(row[mapping.env_seed], f"{where}, {mapping.env_seed}"),
                    policy_seed=policy_seed,
                    simulator=simulator,
                    physics_hash=physics_hash,
                    success=_as_bool(row[mapping.success], f"{where}, {mapping.success}"),
                    failure_reason=failure_reason,
                    metrics=_metrics(row, metric_columns, where),
                    source="tabular",
                    has_timeseries=False,
                    started_at=now,
                )
            )
        batch = Batch(
            batch_id=batch_id,
            policy_version=policy_version,
            suite_ref=suite.ref,
            created_at=now,
            deterministic=detect_determinism(runs),
            n_runs=len(runs),
        )
        return BatchData(batch=batch, runs=runs)


def _metrics(row: dict[str, Any], columns: list[str], where: str) -> dict[str, float]:
    """Empty and NaN cells are left out (the measure is undefined for that run)."""
    out: dict[str, float] = {}
    for column in columns:
        value = row[column]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ReaderError(f"{where}, {column}: expected a number, got {value!r}")
        number = float(value)
        if math.isnan(number):
            continue
        if math.isinf(number):
            raise ReaderError(f"{where}, {column}: infinite value")
        out[column] = number
    return out
