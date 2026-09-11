"""Tests for exhaustive enumeration, the ground truth every other solver is scored on.

Exhaustive search is correct by construction: it evaluates every turn sequence, keeps
the self-avoiding ones and takes the minimum. What needs testing is that it enumerates
the space it claims to, agrees with an independent brute-force calculation, and returns
a conformation that actually decodes to the energy it reports.
"""

from __future__ import annotations

import itertools

import pytest

from foldq.hamiltonian import build_hamiltonian
from foldq.lattice import CubicLattice, TetrahedralLattice
from foldq.peptide import FoldingInstance, Peptide, hp_energy
from foldq.solvers.exact import ExactSolver, exhaustive_search

SEQUENCES = ("HPHPPH", "HHPPHH", "HPPHPPH", "HHPHPPHH")


@pytest.mark.parametrize("sequence", SEQUENCES)
def test_matches_an_independent_brute_force(sequence: str) -> None:
    """The solver's optimum equals a direct sweep written without its machinery.

    The reference loop here deliberately shares nothing with the solver beyond the
    lattice and the energy function, so a bug in the solver's enumeration or symmetry
    handling shows up as a disagreement.
    """
    lattice = TetrahedralLattice()
    peptide = Peptide(sequence)
    reference = min(
        hp_energy(peptide, lattice.contact_pairs(lattice.walk(turns)))
        for turns in itertools.product(range(4), repeat=peptide.length - 1)
        if lattice.is_self_avoiding(turns)
    )
    result = exhaustive_search(FoldingInstance(peptide, lattice))
    assert result.energy == reference


@pytest.mark.parametrize("sequence", SEQUENCES)
def test_returned_conformation_decodes_to_the_reported_energy(sequence: str) -> None:
    """The winning turn sequence is self-avoiding and re-scores to the same energy.

    A solver that reports an energy it cannot produce a conformation for is reporting a
    number, not a solution.
    """
    lattice = TetrahedralLattice()
    peptide = Peptide(sequence)
    result = exhaustive_search(FoldingInstance(peptide, lattice))
    assert result.turns is not None
    assert lattice.is_self_avoiding(result.turns)
    positions = lattice.walk(result.turns)
    assert hp_energy(peptide, lattice.contact_pairs(positions)) == result.energy


def test_symmetry_fixing_does_not_change_the_reported_optimum() -> None:
    """Searching the reduced space finds the same energy as searching all of it."""
    lattice = TetrahedralLattice()
    for sequence in SEQUENCES:
        instance = FoldingInstance(Peptide(sequence), lattice)
        reduced = exhaustive_search(instance, fix_symmetry=True)
        full = exhaustive_search(instance, fix_symmetry=False)
        assert reduced.energy == full.energy, f"{sequence=}"
        assert reduced.n_evaluations < full.n_evaluations


def test_all_hydrophobic_chain_has_a_strictly_negative_optimum() -> None:
    """A chain with enough hydrophobic residues can always form a contact.

    A sanity check with a sign: if the energy came back zero or positive for an
    all-H chain long enough to fold back on itself, the contact detection would be
    broken in a way the equality tests above could not reveal.
    """
    lattice = TetrahedralLattice()
    result = exhaustive_search(FoldingInstance(Peptide("HHHHHHHH"), lattice))
    assert result.energy < 0


def test_all_polar_chain_has_zero_energy() -> None:
    """With no hydrophobic residues there are no scored contacts, so E = 0."""
    lattice = TetrahedralLattice()
    result = exhaustive_search(FoldingInstance(Peptide("PPPPPP"), lattice))
    assert result.energy == 0


def test_works_on_the_cubic_validation_lattice_too() -> None:
    """The solver is not tied to the tetrahedral lattice.

    The cubic lattice is the geometry validated against published walk counts, so being
    able to run the same search there is what lets those two checks meet.
    """
    lattice = CubicLattice()
    peptide = Peptide("HPPHPH")
    reference = min(
        hp_energy(peptide, lattice.contact_pairs(lattice.walk(turns)))
        for turns in itertools.product(range(6), repeat=peptide.length - 1)
        if lattice.is_self_avoiding(turns)
    )
    result = exhaustive_search(FoldingInstance(peptide, lattice), fix_symmetry=False)
    assert result.energy == reference


def test_solver_protocol_is_satisfied_and_results_are_reproducible() -> None:
    """ExactSolver conforms to the Solver protocol and is deterministic.

    Exhaustive search ignores its seed because it is not stochastic, but it must still
    accept one so the benchmark runner can treat every solver identically.
    """
    solver = ExactSolver()
    instance = FoldingInstance(Peptide("HPHPPH"), TetrahedralLattice())
    hamiltonian = build_hamiltonian(instance)
    first = solver.solve(hamiltonian, seed=0)
    second = solver.solve(hamiltonian, seed=99)
    assert solver.name == "exact"
    assert first.energy == second.energy
    assert first.turns == second.turns
    assert first.n_evaluations == second.n_evaluations


def test_evaluation_count_matches_the_reduced_search_space() -> None:
    """The reported evaluation count is the number of turn sequences examined.

    With two turns fixed there are 4**(n_turns - 2) candidates, and time-to-solution
    comparisons in M6 depend on this count meaning exactly that.
    """
    instance = FoldingInstance(Peptide("HPHPPH"), TetrahedralLattice())
    result = exhaustive_search(instance, fix_symmetry=True)
    assert result.n_evaluations == 4 ** (instance.n_turns - 2)
