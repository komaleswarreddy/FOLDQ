"""Solver backends.

Every solver implements the structural interface in :mod:`foldq.solvers.base` and
returns a :class:`~foldq.solvers.base.SolverResult`, so the benchmark runner can gain a
solver without being edited.
"""

from foldq.solvers.base import Solver, SolverResult
from foldq.solvers.exact import ExactSolver, exhaustive_search

__all__ = ["ExactSolver", "Solver", "SolverResult", "exhaustive_search"]
