from __future__ import annotations

from pathlib import Path

import pytest

from polcheck.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / ".polcheck")
