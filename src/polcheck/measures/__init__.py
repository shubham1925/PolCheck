"""Measure registry, the `@measure` decorator and plug-in loading (BUILD_PLAN 7.3).

A measure turns one recorded run into one number (or None when it is undefined
for that run). Measures come from four places:

- built-in (`polcheck.measures.builtin`),
- installed packages, via the `polcheck.measures` entry-point group (the entry
  point names a module, which is scanned for `@measure` functions, or a single
  decorated function),
- Python files listed in `measure_paths` in polcheck.toml,
- imported metrics: numbers that arrived with the runs (e.g. a CSV column),
  which become measures once polcheck.toml gives them a direction and threshold.

`[measures.<name>]` in polcheck.toml overrides threshold, role and direction.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from polcheck.config import Config
from polcheck.schema import Run, Scenario

ENTRY_POINT_GROUP = "polcheck.measures"
CONTACTS = "contacts"
"""Special `requires` entry: the run must have a contacts table (possibly empty)."""

Worse = Literal["higher", "lower"]
Role = Literal["gate", "advisory"]
Kind = Literal["success_rate", "per_run", "imported"]
Source = Literal["builtin", "entry_point", "file", "imported"]

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_ATTR = "__polcheck_measure__"


class MeasureError(ValueError):
    """A measure is defined or configured incorrectly."""


# --- What a measure sees ----------------------------------------------------


class TimeSeries:
    """A run's time-series, looked up by signal name.

    `ts["clearance"]` is a 1-D array; a prefix such as `ts["ee_pos"]` or
    `ts["obj.box.pos"]` returns the matching columns stacked as an (n, k) array
    in recorded order. Custom signals are under `sig.<name>`.
    """

    def __init__(self, table: pa.Table | None) -> None:
        self._table = table
        self._names: list[str] = table.column_names if table is not None else []

    def columns(self, prefix: str) -> list[str]:
        if prefix in self._names:
            return [prefix]
        return [n for n in self._names if n.startswith(prefix + ".")]

    def has(self, prefix: str) -> bool:
        return bool(self.columns(prefix))

    def __getitem__(self, prefix: str) -> NDArray[np.float64]:
        cols = self.columns(prefix)
        if self._table is None or not cols:
            raise KeyError(f"no signal {prefix!r} in this run")
        arrays: list[NDArray[np.float64]] = [
            np.asarray(self._table.column(c).to_numpy(), dtype=np.float64) for c in cols
        ]
        if cols == [prefix]:
            return arrays[0]
        return np.column_stack(arrays)

    @property
    def t(self) -> NDArray[np.float64]:
        return self["t"]

    def __len__(self) -> int:
        return self._table.num_rows if self._table is not None else 0


@dataclass(frozen=True)
class RunData:
    """Everything a measure may use about one run."""

    run: Run
    scenario: Scenario | None
    """The run's scenario from the suite (None if the suite no longer has it)."""
    ts: TimeSeries
    contacts_table: pa.Table | None
    """Contacts, or None if the run did not track contacts."""

    @property
    def contacts(self) -> pa.Table:
        if self.contacts_table is None:
            raise KeyError("this run did not track contacts")
        return self.contacts_table


MeasureFunc = Callable[[RunData], float | None]


# --- Measure definitions ----------------------------------------------------


@dataclass(frozen=True)
class MeasureSpec:
    name: str
    kind: Kind
    worse: Worse | None
    threshold: float | None
    """Smallest change in the measure's own units that matters."""
    role: Role
    unit: str = ""
    success_only: bool = False
    requires: tuple[str, ...] = ()
    version: str = "1"
    """Bump when the definition changes, so cached values are recomputed."""
    func: MeasureFunc | None = None
    source: Source = "builtin"
    origin: str = ""
    """Where it was defined (module, file or "polcheck.toml")."""
    description: str = ""

    @property
    def configured(self) -> bool:
        """Has a direction and threshold, so it can take part in the verdict."""
        return self.worse is not None and self.threshold is not None

    @property
    def cache_key(self) -> str:
        return f"{self.name}@{self.version}"


def _check_threshold(name: str, threshold: float | None) -> None:
    if threshold is not None and not (math.isfinite(threshold) and threshold > 0):
        raise MeasureError(f"measure {name!r}: threshold must be a positive number")


def measure(
    *,
    name: str,
    worse: Worse,
    threshold: float,
    unit: str = "",
    requires: Iterable[str] = (),
    role: Role = "advisory",
    success_only: bool = False,
    version: str = "1",
) -> Callable[[MeasureFunc], MeasureFunc]:
    """Declare a per-run measure.

    The function receives a `RunData` and returns a float, or None when the
    measure is undefined for that run. It is only called when every signal in
    `requires` is present (use `"contacts"` to require a contacts table).
    """
    if not _NAME.match(name):
        raise MeasureError(
            f"measure name {name!r} must start with a letter and contain only letters, "
            "digits and underscores"
        )
    if worse not in ("higher", "lower"):
        raise MeasureError(f"measure {name!r}: worse must be 'higher' or 'lower'")
    if role not in ("gate", "advisory"):
        raise MeasureError(f"measure {name!r}: role must be 'gate' or 'advisory'")
    _check_threshold(name, threshold)

    def decorate(func: MeasureFunc) -> MeasureFunc:
        spec = MeasureSpec(
            name=name,
            kind="per_run",
            worse=worse,
            threshold=float(threshold),
            role=role,
            unit=unit,
            success_only=success_only,
            requires=tuple(requires),
            version=str(version),
            func=func,
            description=(func.__doc__ or "").strip().split("\n")[0],
        )
        setattr(func, _ATTR, spec)
        return func

    return decorate


SUCCESS_RATE = MeasureSpec(
    name="success_rate",
    kind="success_rate",
    worse="lower",
    threshold=0.03,
    role="gate",
    unit="fraction",
    origin="polcheck",
    description="Share of runs that succeeded; tested on the batch, not per run.",
)


# --- Registry ---------------------------------------------------------------


def _scan(module: ModuleType, source: Source) -> list[MeasureSpec]:
    """Measures defined in `module` itself (not ones it merely imports)."""
    found = []
    for value in vars(module).values():
        spec = getattr(value, _ATTR, None)
        if isinstance(spec, MeasureSpec) and getattr(value, "__module__", None) == module.__name__:
            found.append(replace(spec, source=source, origin=module.__name__))
    return found


def _load_file(path: Path) -> ModuleType:
    if not path.is_file():
        raise MeasureError(f"measure file not found: {path}")
    module_name = f"polcheck_user_measures_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise MeasureError(f"cannot load measures from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except MeasureError:
        raise
    except Exception as exc:
        raise MeasureError(f"error while loading measures from {path}: {exc}") from exc
    return module


class Registry:
    """All known measures, with polcheck.toml overrides applied."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._specs: dict[str, MeasureSpec] = {}

    def add(self, spec: MeasureSpec) -> None:
        existing = self._specs.get(spec.name)
        if existing is not None:
            raise MeasureError(
                f"measure {spec.name!r} is defined twice: in {existing.origin} "
                f"({existing.source}) and in {spec.origin} ({spec.source})"
            )
        self._specs[spec.name] = self._apply_config(spec)

    def _apply_config(self, spec: MeasureSpec) -> MeasureSpec:
        override = self.config.measures.get(spec.name)
        if override is None:
            return spec
        _check_threshold(spec.name, override.threshold)
        return replace(
            spec,
            worse=override.worse or spec.worse,
            threshold=override.threshold if override.threshold is not None else spec.threshold,
            role=override.role or spec.role,
        )

    def __getitem__(self, name: str) -> MeasureSpec:
        return self._specs[name]

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def specs(self) -> list[MeasureSpec]:
        return list(self._specs.values())

    def per_run(self) -> list[MeasureSpec]:
        return [s for s in self._specs.values() if s.kind == "per_run"]

    def imported(self, metric_names: Iterable[str]) -> list[MeasureSpec]:
        """Measures for imported metrics not already registered under that name.

        Those without a direction and threshold in polcheck.toml come back
        unconfigured and are reported but left out of the verdict.
        """
        out = []
        for name in sorted(set(metric_names) - self._specs.keys()):
            override = self.config.measures.get(name)
            out.append(
                MeasureSpec(
                    name=name,
                    kind="imported",
                    worse=override.worse if override else None,
                    threshold=override.threshold if override else None,
                    role=(override.role if override and override.role else "advisory"),
                    source="imported",
                    origin="imported metric",
                )
            )
        return out

    def config_only(self) -> list[str]:
        """Names configured in polcheck.toml that no registered measure has.

        Expected for imported metrics; a typo otherwise.
        """
        return sorted(set(self.config.measures) - self._specs.keys())


def load_registry(config: Config) -> Registry:
    """Built-ins, then entry points, then `measure_paths` files."""
    from polcheck.measures import builtin

    registry = Registry(config)
    registry.add(SUCCESS_RATE)
    for spec in _scan(builtin, "builtin"):
        registry.add(spec)
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            obj: Any = ep.load()
        except Exception as exc:
            raise MeasureError(f"cannot load measure entry point {ep.name!r}: {exc}") from exc
        if isinstance(obj, ModuleType):
            specs = _scan(obj, "entry_point")
        elif isinstance(getattr(obj, _ATTR, None), MeasureSpec):
            specs = [replace(getattr(obj, _ATTR), source="entry_point", origin=ep.value)]
        else:
            raise MeasureError(
                f"entry point {ep.name!r} ({ep.value}) is neither a module nor an @measure function"
            )
        for spec in specs:
            registry.add(spec)
    for path in config.measure_paths:
        for spec in _scan(_load_file(path), "file"):
            registry.add(replace(spec, origin=str(path)))
    return registry
