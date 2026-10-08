"""polcheck command-line interface (BUILD_PLAN 7.8).

Exit codes: 0 PASS, 1 REGRESSED, 2 INCONCLUSIVE, 3 not comparable, 4 usage or
internal error. Click's own usage errors (exit 2) and uncaught exceptions
(exit 1) would collide with verdict codes, so `main` maps both to 4.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from polcheck.config import ConfigError, load_config
from polcheck.measures import MeasureError, MeasureSpec, load_registry
from polcheck.readers import ReaderError, get_reader
from polcheck.schema import UNKNOWN, SimulatorInfo
from polcheck.store import Store, StoreError
from polcheck.suite import SuiteError

EXIT_USAGE = 4

app = typer.Typer(
    name="polcheck",
    help="Regression testing for robot policies.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", help="Path to polcheck.toml (default: ./polcheck.toml if present)."),
]

_USER_ERRORS = (ConfigError, MeasureError, ReaderError, StoreError, SuiteError)


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(EXIT_USAGE)


@app.command()
def reindex(config: ConfigOption = None) -> None:
    """Rebuild the store's index.duckdb from its Parquet files."""
    try:
        cfg = load_config(config)
        n = Store(cfg.store).reindex()
    except _USER_ERRORS as exc:
        _fail(str(exc))
    typer.echo(f"Reindexed {n} batches in {cfg.store}")


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="File to import (e.g. per-episode CSV or JSONL).")],
    reader: Annotated[str, typer.Option("--reader", help="Reader name, e.g. tabular.")],
    policy: Annotated[str, typer.Option("--policy", help="Policy version, e.g. pick-v14.")],
    suite: Annotated[
        str,
        typer.Option(
            "--suite", help="NAME@VER of a registered suite, or a path to the suite TOML."
        ),
    ],
    simulator_name: Annotated[str | None, typer.Option(help="Overrides the mapping.")] = None,
    simulator_version: Annotated[str | None, typer.Option(help="Overrides the mapping.")] = None,
    physics_hash: Annotated[str | None, typer.Option(help="Overrides the mapping.")] = None,
    config: ConfigOption = None,
) -> None:
    """Import a batch of runs from another tool's output."""
    try:
        cfg = load_config(config)
        mapping = cfg.readers.tabular
        overrides: dict[str, object] = {}
        if simulator_name or simulator_version:
            current = mapping.simulator or SimulatorInfo(name=UNKNOWN, version=UNKNOWN)
            overrides["simulator"] = SimulatorInfo(
                name=simulator_name or current.name, version=simulator_version or current.version
            )
        if physics_hash:
            overrides["physics_hash"] = physics_hash
        if overrides:
            readers = cfg.readers.model_copy(
                update={"tabular": mapping.model_copy(update=overrides)}
            )
            cfg = cfg.model_copy(update={"readers": readers})
        store = Store(cfg.store)
        suite_model = store.resolve_suite(suite)
        data = get_reader(reader).read(path, policy_version=policy, suite=suite_model, config=cfg)
        store.write_batch(data)
    except _USER_ERRORS as exc:
        _fail(str(exc))
    typer.echo(
        f"Ingested {data.batch.n_runs} runs as batch {data.batch.batch_id} "
        f"(policy {policy}, suite {suite_model.ref})"
    )


measures_app = typer.Typer(help="Inspect measures.", no_args_is_help=True)
app.add_typer(measures_app, name="measures")


def _measure_row(spec: MeasureSpec) -> list[str]:
    threshold = "-" if spec.threshold is None else f"{spec.threshold:g}"
    unit = f" {spec.unit}" if spec.unit and spec.threshold is not None else ""
    worse = f"{spec.worse} is worse" if spec.worse else "-"
    notes = []
    if spec.success_only:
        notes.append("successful runs only")
    if not spec.configured:
        notes.append("unconfigured: set worse and threshold to use it")
    return [spec.name, worse, threshold + unit, spec.role, spec.source, "; ".join(notes)]


@measures_app.command("list")
def measures_list(config: ConfigOption = None) -> None:
    """List every measure with its direction, threshold, role and source."""
    try:
        cfg = load_config(config)
        registry = load_registry(cfg)
    except _USER_ERRORS as exc:
        _fail(str(exc))
    specs = registry.specs() + registry.imported(registry.config_only())
    rows = [["MEASURE", "DIRECTION", "THRESHOLD", "ROLE", "SOURCE", "NOTES"]]
    rows += [_measure_row(spec) for spec in specs]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        typer.echo("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip())
    imported = [s.name for s in specs if s.source == "imported"]
    if imported:
        typer.echo(
            f"\nImported metrics configured in polcheck.toml: {', '.join(imported)}. They "
            "are compared only if a batch's runs carry a metric with that name."
        )


def main() -> None:
    try:
        # Non-standalone mode returns typer.Exit codes and raises usage errors to us.
        result = app(standalone_mode=False)
    except typer.TyperException as exc:  # Click usage errors (missing option, bad value)
        if message := exc.format_message():  # empty when help was shown for no arguments
            typer.echo(f"error: {message}", err=True)
            typer.echo("Try 'polcheck --help' for help.", err=True)
        sys.exit(EXIT_USAGE)
    except typer.Abort:
        typer.echo("Aborted.", err=True)
        sys.exit(EXIT_USAGE)
    except Exception:
        traceback.print_exc()
        typer.echo("error: internal error (see traceback above)", err=True)
        sys.exit(EXIT_USAGE)
    sys.exit(result if isinstance(result, int) else 0)
