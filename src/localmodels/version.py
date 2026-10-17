"""This package's installed version, from its metadata (pyproject.toml is the one place it is written)."""

from __future__ import annotations

from functools import cache
from importlib.metadata import PackageNotFoundError, version


@cache
def package_version() -> str:
    """The installed version of localmodels; ``0+unknown`` from a source tree that was never installed."""
    try:
        return version("localmodels")
    except PackageNotFoundError:
        return "0+unknown"
