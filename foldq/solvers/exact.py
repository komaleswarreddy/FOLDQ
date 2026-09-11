"""Exhaustive enumeration: the ground truth the benchmark is scored against.

Every turn sequence is generated, the self-avoiding ones are scored, and the minimum is
taken. There is no approximation anywhere in this module, which is the point: success
probability and approximation ratio in M6 are only meaningful because the exact optimum
is known independently.

Cost is ``4**(N-1)`` sequences before symmetry fixing and ``4**(N-3)`` after, so the
approach is feasible to roughly ``N = 10`` -- 16384 candidates for the reduced search,
which runs in well under a second.
"""

from __future__ import annotations

import time

from foldq.encoding import enumerate_turns
from foldq.peptide import FoldingInstance, hp_energy
from foldq.solvers.base import SolverResult


def exhaustive_search(
    instance: FoldingInstance, *, fix_symmetry: bool = True
) -> SolverResult:
    """Enumerate every conformation and return the lowest-energy one.

    Parameters
    ----------
    instance
        The folding instance to solve.
    fix_symmetry
        When true, search only the symmetry-reduced space (see
        :data:`foldq.encoding.FIXED_TURNS`). The optimum is unchanged; the search is
        twelve times smaller. Symmetry fixing is specific to the tetrahedral lattice,
        so it is ignored for any other lattice.

    Notes
    -----
    ``n_evaluations`` counts candidate turn sequences examined, including those
    rejected as self-intersecting. Rejection costs a walk, so it is real work and
    excluding it would flatter this solver against the stochastic ones.
    """
    lattice = instance.lattice
    reduce_symmetry = fix_symmetry and lattice.name == "tetrahedral"

    if reduce_symmetry:
        candidates = enumerate_turns(instance.n_turns, fix_symmetry=True)
    else:
        candidates = enumerate_turns(instance.n_turns, fix_symmetry=False)

    best_energy: float | None = None
    best_turns: tuple[int, ...] | None = None
    evaluations = 0

    start = time.perf_counter()
    for turns in candidates:
        evaluations += 1
        positions = lattice.walk(turns)
        if len(set(positions)) != len(positions):
            continue
        energy = hp_energy(instance.peptide, lattice.contact_pairs(positions))
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_turns = turns
    elapsed = time.perf_counter() - start

    if best_energy is None:
        message = (
            f"no self-avoiding conformation exists for a chain of "
            f"{instance.n_beads} beads on the {lattice.name} lattice"
        )
        raise ValueError(message)

    return SolverResult(
        energy=best_energy,
        turns=best_turns,
        n_evaluations=evaluations,
        wall_time_s=elapsed,
        seed=None,
        metadata={
            "lattice": lattice.name,
            "symmetry_fixed": reduce_symmetry,
            "exact": True,
        },
    )


class ExactSolver:
    """Exhaustive enumeration, wrapped in the :class:`~foldq.solvers.base.Solver` API.

    Deterministic, so the seed is accepted and recorded but does not influence the
    search. It is accepted anyway so the benchmark runner can call every solver the
    same way.
    """

    def __init__(self, *, fix_symmetry: bool = True) -> None:
        """Store whether to search the symmetry-reduced space."""
        self._fix_symmetry = fix_symmetry

    @property
    def name(self) -> str:
        """Short identifier used in artifacts, tables and figures."""
        return "exact"

    def solve(self, instance: FoldingInstance, seed: int | None = None) -> SolverResult:
        """Return the exact optimum for ``instance``; ``seed`` is ignored."""
        result = exhaustive_search(instance, fix_symmetry=self._fix_symmetry)
        return SolverResult(
            energy=result.energy,
            turns=result.turns,
            n_evaluations=result.n_evaluations,
            wall_time_s=result.wall_time_s,
            seed=seed,
            metadata=result.metadata,
        )
