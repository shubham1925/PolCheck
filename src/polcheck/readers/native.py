"""Reads batches written by the native recorder."""

from __future__ import annotations

from pathlib import Path

from polcheck.config import Config
from polcheck.readers.base import ReaderError
from polcheck.schema import BatchData, Suite
from polcheck.store import StoreError, read_batch_dir


class NativeReader:
    name = "native"

    def read(self, path: Path, *, policy_version: str, suite: Suite, config: Config) -> BatchData:
        """`path` is a batch directory (`<store>/batches/<batch_id>`)."""
        try:
            data = read_batch_dir(path)
        except StoreError as exc:
            raise ReaderError(str(exc)) from exc
        if data.batch.suite_ref != suite.ref:
            raise ReaderError(
                f"{path} was recorded on suite {data.batch.suite_ref}, not {suite.ref}"
            )
        if data.batch.policy_version != policy_version:
            raise ReaderError(
                f"{path} holds policy version {data.batch.policy_version!r}, not {policy_version!r}"
            )
        return data
