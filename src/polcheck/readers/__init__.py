"""Readers turn recorded or imported results into a `BatchData` (BUILD_PLAN 7.2)."""

from polcheck.readers.base import Reader, ReaderError, get_reader

__all__ = ["Reader", "ReaderError", "get_reader"]
