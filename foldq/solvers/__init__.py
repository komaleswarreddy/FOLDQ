"""Solver backends.

Every solver implements the structural interface in :mod:`foldq.solvers.base` and
returns a :class:`~foldq.solvers.base.SolverResult`, so the benchmark runner can gain a
solver without being edited.
"""

from foldq.solvers.annealing import AnnealingConfig, AnnealingSolver, anneal
from foldq.solvers.base import Solver, SolverResult
from foldq.solvers.bifurcation import (
    BifurcationConfig,
    BifurcationSolver,
    ballistic_bifurcation,
)
from foldq.solvers.exact import ExactSolver, exhaustive_search

__all__ = [
    "AnnealingConfig",
    "AnnealingSolver",
    "BifurcationConfig",
    "BifurcationSolver",
    "ExactSolver",
    "Solver",
    "SolverResult",
    "anneal",
    "ballistic_bifurcation",
    "exhaustive_search",
]
