"""The solver interface every backend implements.

The benchmark runner must be able to add a solver without being edited, so solvers are
structural implementations of :class:`Solver` rather than subclasses of a base class,
and every one of them returns the same :class:`SolverResult`.

Every solver takes an explicit seed, including the ones that are deterministic. A
uniform signature is what lets the runner treat them identically, and Guardrail 4
requires that two runs with the same seed produce identical output.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from foldq.hamiltonian import FoldingHamiltonian


@dataclass(frozen=True)
class SolverResult:
    """What a solver reports about one run.

    Attributes
    ----------
    energy
        Lowest energy found. Comparable across solvers only because every solver is
        handed the identical instance.
    turns
        The turn sequence achieving ``energy``, or ``None`` if the solver found no
        feasible conformation at all. A reported energy without a conformation that
        reproduces it is a number, not a solution, so callers are expected to check.
    n_evaluations
        Objective-function evaluations performed. The unit of work for comparing
        solvers independently of machine speed.
    wall_time_s
        Elapsed wall-clock time for the run, in seconds. Needed for time-to-solution
        in M6, which wall-clock alone cannot express for stochastic solvers.
    seed
        The seed the run was given, recorded so the run can be reproduced exactly.
    metadata
        Solver-specific detail, kept out of the common fields so that adding a solver
        never changes this dataclass.
    """

    energy: float
    turns: tuple[int, ...] | None
    n_evaluations: int
    wall_time_s: float
    seed: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class Solver(Protocol):
    """Structural interface for anything that can solve a folding instance."""

    @property
    def name(self) -> str:
        """Short identifier used in artifacts, tables and figures."""
        ...

    def solve(
        self, hamiltonian: FoldingHamiltonian, seed: int | None = None
    ) -> SolverResult:
        """Search ``hamiltonian`` and report the best conformation found.

        Every solver receives the Hamiltonian rather than the raw instance, so the
        benchmark can guarantee they were all given the identical problem. Solvers that
        do not need the encoded form, such as exhaustive enumeration, read
        ``hamiltonian.instance``.
        """
        ...
