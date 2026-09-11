"""Smoke tests for the M0 scaffold.

These assert that the package is installed, importable, typed, and that its declared
version agrees with the version recorded in the distribution metadata. A disagreement
almost always means a stale editable install, which silently invalidates every later
benchmark artifact that records package versions for reproducibility.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import foldq


def test_package_is_importable() -> None:
    """The package imports and exposes a non-empty version string."""
    assert isinstance(foldq.__version__, str)
    assert foldq.__version__


def test_source_version_matches_distribution_metadata() -> None:
    """``foldq.__version__`` agrees with the installed distribution metadata.

    Guardrail 1 requires every published number to trace to a reproducible
    environment. Benchmark artifacts record package versions, so a mismatch between
    the source tree and what is actually installed would make those records wrong.
    """
    assert foldq.__version__ == importlib.metadata.version("foldq")


def test_package_ships_py_typed_marker() -> None:
    """A PEP 561 ``py.typed`` marker ships with the package.

    Without it, downstream ``mypy`` runs silently treat ``foldq`` as untyped rather
    than failing loudly, which defeats the strict type checking configured here.
    """
    marker = Path(foldq.__file__).parent / "py.typed"
    assert marker.is_file()
