"""Compute a batch's measure values, using imported values and the cache.

For each per-run measure, in order of preference:

1. **imported:** if the batch's runs carry a metric with the measure's name, the
   evaluation tool already measured it and its values are used (the plan's rule:
   never re-measure what the customer's tool provides);
2. **cached:** values already in `measures.parquet` for this measure version;
3. **computed:** call the measure on each run and cache the result.

A run gets no value (None, with a reason) when the measure is success-only and
the run failed, a required signal is missing, or the measure returns nothing,
returns a non-finite number or raises. Measures never stop a comparison.

With `resample_hz` set (two batches logged at different rates, DECISIONS.md D31),
rate-sensitive measures are computed on time-series resampled to that rate and
cached under their own `<name>@<version>@<rate>Hz` column.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

import pyarrow as pa

from polcheck.measures import CONTACTS, MeasureSpec, Registry, RunData, TimeSeries
from polcheck.measures.resample import resample_table
from polcheck.schema import BatchData, Run, Scenario, Suite
from polcheck.store import Store
from polcheck.suite import scenario_id

log = logging.getLogger(__name__)

SKIP_SUFFIX = "#skip"
"""Cache column holding the reason a run has no value (`<name>@<version>#skip`)."""

FAILED_RUN = "run failed (measured on successful runs only)"


@dataclass
class MeasureValues:
    spec: MeasureSpec
    origin: Literal["computed", "cached", "imported"]
    values: dict[UUID, float | None] = field(default_factory=dict)
    skipped: dict[UUID, str] = field(default_factory=dict)
    """Why each run without a value has none."""
    resampled_hz: float | None = None
    """Set when the values were computed on time-series resampled to this rate."""
    notes: list[str] = field(default_factory=list)

    def defined(self) -> dict[UUID, float]:
        return {k: v for k, v in self.values.items() if v is not None}


class _RunLoader:
    """Loads each run's time-series and contacts once, on first use."""

    def __init__(
        self,
        store: Store,
        data: BatchData,
        scenarios: dict[str, Scenario],
        resample_hz: float | None = None,
    ) -> None:
        self._store, self._batch_id, self._scenarios = store, data.batch.batch_id, scenarios
        self._resample_hz = resample_hz
        self._cache: dict[UUID, RunData] = {}
        self.not_filtered = 0
        """Runs too short for the low-pass filter (resampling only)."""

    def __call__(self, run: Run) -> RunData:
        if run.run_id not in self._cache:
            table = self._store.read_timeseries(self._batch_id, run.run_id)
            if table is not None and self._resample_hz is not None:
                table, note = resample_table(table, self._resample_hz)
                self.not_filtered += note is not None
            self._cache[run.run_id] = RunData(
                run=run,
                scenario=self._scenarios.get(run.scenario_id),
                ts=TimeSeries(table),
                contacts_table=self._store.read_contacts(self._batch_id, run.run_id),
            )
        return self._cache[run.run_id]


def _missing(spec: MeasureSpec, data: RunData) -> list[str]:
    return [
        req
        for req in spec.requires
        if not (data.contacts_table is not None if req == CONTACTS else data.ts.has(req))
    ]


def _compute_one(spec: MeasureSpec, data: RunData) -> tuple[float | None, str | None]:
    if spec.success_only and not data.run.success:
        return None, FAILED_RUN
    missing = _missing(spec, data)
    if missing:
        return None, f"missing signal(s): {', '.join(missing)}"
    assert spec.func is not None
    try:
        value = spec.func(data)
    except Exception as exc:
        return None, f"error: {type(exc).__name__}: {exc}"
    if value is None:
        return None, "measure returned no value"
    value = float(value)
    if not math.isfinite(value):
        return None, f"non-finite value ({value})"
    return value, None


def _imported(spec: MeasureSpec, runs: Sequence[Run]) -> MeasureValues:
    result = MeasureValues(spec, "imported")
    for run in runs:
        value = run.metrics.get(spec.name)
        reason: str | None
        if spec.success_only and not run.success:
            value, reason = None, FAILED_RUN
        else:
            reason = None if value is not None else "not in the imported data"
        result.values[run.run_id] = value
        if reason:
            result.skipped[run.run_id] = reason
    return result


def _from_cache(spec: MeasureSpec, cache: pa.Table, key: str) -> MeasureValues:
    result = MeasureValues(spec, "cached")
    ids = cache.column("run_id").to_pylist()
    values = cache.column(key).to_pylist()
    reasons = cache.column(key + SKIP_SUFFIX).to_pylist()
    for run_id, value, reason in zip(ids, values, reasons, strict=True):
        result.values[UUID(run_id)] = value
        if reason is not None:
            result.skipped[UUID(run_id)] = reason
    return result


def _name_and_rate(column: str) -> tuple[str, str | None]:
    """Split a cache column `<name>@<version>[@<rate>Hz][#skip]` into name and rate tag."""
    parts = column.removesuffix(SKIP_SUFFIX).split("@")
    return parts[0], parts[2] if len(parts) > 2 else None


def _write_cache(
    store: Store,
    data: BatchData,
    cache: pa.Table | None,
    computed: dict[str, MeasureValues],
) -> None:
    """Add the computed columns (keyed by cache column name) to the batch's cache."""
    run_ids = [str(r.run_id) for r in data.runs]
    columns: dict[str, pa.Array] = {"run_id": pa.array(run_ids, pa.string())}
    # Keep everything except older versions of the same measure at the same rate.
    replaced = {_name_and_rate(key) for key in computed}
    if cache is not None and cache.column("run_id").to_pylist() == run_ids:
        for name in cache.column_names:
            if name != "run_id" and _name_and_rate(name) not in replaced:
                columns[name] = cache.column(name)
    for key, m in computed.items():
        columns[key] = pa.array([m.values[r.run_id] for r in data.runs], pa.float64())
        columns[key + SKIP_SUFFIX] = pa.array(
            [m.skipped.get(r.run_id) for r in data.runs], pa.string()
        )
    store.write_measure_cache(data.batch.batch_id, pa.table(columns))


def measure_batch(
    store: Store,
    data: BatchData,
    suite: Suite,
    registry: Registry,
    *,
    resample_hz: float | None = None,
) -> dict[str, MeasureValues]:
    """Values of every per-run and imported measure for one batch, keyed by name."""
    runs = data.runs
    metric_names = {name for r in runs for name in r.metrics}
    scenarios = {scenario_id(sc): sc for sc in suite.scenarios}
    load = _RunLoader(store, data, scenarios)
    load_resampled = _RunLoader(store, data, scenarios, resample_hz)

    cache = store.read_measure_cache(data.batch.batch_id)
    cache_ok = cache is not None and cache.column("run_id").to_pylist() == [
        str(r.run_id) for r in runs
    ]
    results: dict[str, MeasureValues] = {}
    computed: dict[str, MeasureValues] = {}

    for spec in registry.per_run():
        resampled = resample_hz is not None and spec.rate_sensitive
        key = spec.cache_key_at(resample_hz)
        if spec.name in metric_names:
            result = _imported(spec, runs)
        elif cache_ok and cache is not None and key in cache.column_names:
            result = _from_cache(spec, cache, key)
        else:
            result = MeasureValues(spec, "computed")
            loader = load_resampled if resampled else load
            for run in runs:
                value, reason = _compute_one(spec, loader(run))
                result.values[run.run_id] = value
                if reason:
                    result.skipped[run.run_id] = reason
            errors = sum(r.startswith("error:") for r in result.skipped.values())
            if errors:
                log.warning("measure %s raised on %d run(s); see skip reasons", spec.name, errors)
            if resampled and load_resampled.not_filtered:
                result.notes.append(
                    f"{load_resampled.not_filtered} run(s) were too short to low-pass "
                    "and were only interpolated"
                )
            computed[key] = result
        if resampled and result.origin != "imported":
            result.resampled_hz = resample_hz
        results[spec.name] = result

    for spec in registry.imported(metric_names):
        results[spec.name] = _imported(spec, runs)

    if computed:
        _write_cache(store, data, cache if cache_ok else None, computed)
    return results
