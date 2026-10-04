"""Reader protocol and entry-point discovery."""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol

from polcheck.config import Config
from polcheck.schema import BatchData, Suite

ENTRY_POINT_GROUP = "polcheck.readers"


class ReaderError(ValueError):
    pass


class Reader(Protocol):
    name: str

    def read(
        self, path: Path, *, policy_version: str, suite: Suite, config: Config
    ) -> BatchData: ...


def get_reader(name: str) -> Reader:
    """Instantiate the reader registered under `name` in the `polcheck.readers` group."""
    found = {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}
    if name not in found:
        available = ", ".join(sorted(found)) or "none"
        raise ReaderError(f"unknown reader {name!r} (available: {available})")
    reader: Reader = found[name].load()()
    return reader
