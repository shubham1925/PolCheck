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

_USER_ERRORS = (ConfigError, ReaderError, StoreError, SuiteError)


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
