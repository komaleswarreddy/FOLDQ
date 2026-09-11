"""FoldQ: a reproducible benchmark for coarse-grained lattice protein folding.

The package answers one question with data rather than assertion: for peptide folding
at today's hardware scale, how do variational quantum algorithms compare against
classical and quantum-inspired classical solvers on the *identical* problem instance?

Nothing but the package version lives here yet. Each subpackage arrives with the
milestone that implements and tests it -- see ``PROJECT_BRIEF.md``.
"""

__all__ = ["__version__"]

#: Single source of truth for the version. ``pyproject.toml`` reads it from this file
#: via ``[tool.hatch.version]``, so the source and the distribution metadata cannot
#: drift apart; ``tests/test_smoke.py`` asserts that they have not.
__version__ = "0.1.0"
