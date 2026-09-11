"""Tests for the folding Hamiltonian and its three representations.

The Hamiltonian is transcribed from the Supporting Information of Robert et al. (2021),
arXiv:1908.02163. Rather than trusting the transcription, each piece is checked against
something independent: the turn indicators against the turn they claim to select, the
distance polynomial against the lattice geometry of :mod:`foldq.lattice`, the penalty
weights against the bound the paper derives, and the whole Hamiltonian against the
exhaustive optimum from M1.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest

from foldq.encoding import FIXED_TURNS, enumerate_turns
from foldq.hamiltonian import (
    MIN_CONTACT_SEPARATION,
    QubitLayout,
    build_chirality_constraint,
    build_hamiltonian,
    build_layout,
    contact_pairs,
    derive_penalties,
    distance,
    hp_contact_energy,
    turn_indicator,
)
from foldq.lattice import TetrahedralLattice
from foldq.peptide import FoldingInstance, Peptide
from foldq.solvers.exact import exhaustive_search

LATTICE = TetrahedralLattice()


def _instance(sequence: str) -> FoldingInstance:
    """Build a tetrahedral folding instance for an HP sequence."""
    return FoldingInstance(Peptide(sequence), LATTICE)


def _turn_bits(layout: QubitLayout, turns: tuple[int, ...]) -> dict[int, int]:
    """Return the turn-qubit assignment for a turn sequence."""
    return {
        index: (turns[turn] >> 1) if bit == 0 else (turns[turn] & 1)
        for (turn, bit), index in layout.turn_qubits.items()
    }


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def test_turn_indicator_selects_exactly_its_direction() -> None:
    """``f_a(k)`` is 1 when turn ``k`` points along ``a`` and 0 otherwise.

    Checked against every turn assignment, because Eq. SI-3 to SI-6 are written in a
    reduced form that is easy to transcribe subtly wrong.
    """
    instance = _instance("HHHHHH")
    layout = build_layout(instance)
    for turns in enumerate_turns(instance.n_turns, fix_symmetry=True):
        assignment = _turn_bits(layout, turns)
        for k in range(instance.n_turns):
            for a in range(4):
                expected = 1.0 if turns[k] == a else 0.0
                assert turn_indicator(layout, k, a).evaluate(assignment) == expected


def test_indicators_at_each_turn_sum_to_one() -> None:
    """Exactly one direction is selected per turn, so the indicators partition unity."""
    instance = _instance("HHHHHH")
    layout = build_layout(instance)
    for turns in enumerate_turns(instance.n_turns, fix_symmetry=True):
        assignment = _turn_bits(layout, turns)
        for k in range(instance.n_turns):
            total = sum(
                turn_indicator(layout, k, a).evaluate(assignment) for a in range(4)
            )
            assert total == pytest.approx(1.0)


def test_distance_polynomial_matches_lattice_geometry() -> None:
    """``d(i,j)`` agrees with the Euclidean geometry through Eq. SI-14.

    In this package's integer coordinates a bond has squared length 3, and the paper's
    bijection between its turn-counting distance and real distance becomes
    ``squared_euclidean = 4 d - [(j - i) odd]``. Checking the polynomial against the
    independently-implemented lattice walk is what makes the Hamiltonian's notion of
    "in contact" trustworthy.
    """
    instance = _instance("HHHHHHH")
    layout = build_layout(instance)
    for turns in enumerate_turns(instance.n_turns, fix_symmetry=True):
        assignment = _turn_bits(layout, turns)
        positions = LATTICE.walk(turns)
        for i in range(instance.n_beads):
            for j in range(i + 1, instance.n_beads):
                d = distance(layout, i, j).evaluate(assignment)
                predicted = 4 * d - (1 if (j - i) % 2 else 0)
                assert predicted == pytest.approx(
                    LATTICE.squared_distance(positions[i], positions[j])
                ), f"{turns=} {i=} {j=}"


def test_distance_one_means_nearest_neighbour() -> None:
    """``d == 1`` is exactly a first-nearest-neighbour contact."""
    instance = _instance("HHHHHHH")
    layout = build_layout(instance)
    for turns in enumerate_turns(instance.n_turns, fix_symmetry=True):
        assignment = _turn_bits(layout, turns)
        positions = LATTICE.walk(turns)
        for i, j in contact_pairs(instance.n_beads):
            in_contact = LATTICE.are_nearest_neighbours(positions[i], positions[j])
            assert (distance(layout, i, j).evaluate(assignment) == 1) == in_contact


def test_minimum_contact_separation_is_five_on_the_feasible_subspace() -> None:
    """No pair closer than five bonds can touch, once back-tracking is excluded.

    The paper asserts this (SI, Section C) and uses it to restrict the interaction sum,
    which is what keeps the contact register to pairs that can really interact.

    The restriction holds only on the back-track-free subspace. Allowing a turn to
    repeat the previous index lets beads three bonds apart reach contact distance, and
    the assertion below fails. That is not a caveat to wave away: it means the
    contact-register restriction is valid *because* ``H_gc`` forbids back-tracking, so
    a back-track penalty that was too weak would admit contacts the Hamiltonian has no
    ancilla to represent. The two design choices are coupled.
    """
    with_backtracks = set()
    without_backtracks = set()
    for n_turns in range(3, 8):
        for turns in itertools.product(range(4), repeat=n_turns):
            positions = LATTICE.walk(turns)
            back_tracks = any(turns[k] == turns[k + 1] for k in range(n_turns - 1))
            for i in range(len(positions)):
                for j in range(i + 2, len(positions)):
                    if LATTICE.are_nearest_neighbours(positions[i], positions[j]):
                        with_backtracks.add(j - i)
                        if not back_tracks:
                            without_backtracks.add(j - i)

    assert min(without_backtracks) == MIN_CONTACT_SEPARATION
    assert all(separation % 2 == 1 for separation in without_backtracks)
    # And the coupling itself: allowing back-tracks really does open up separation 3.
    assert 3 in with_backtracks
    assert 3 not in without_backtracks


def test_chirality_term_is_empty_for_a_main_chain_model() -> None:
    """``H_ch`` is identically zero, and says so rather than being silently absent.

    Eq. SI-26 constrains side-chain placement. This model has no side chains, so there
    is no stereocentre to constrain.
    """
    layout = build_layout(_instance("HHHHHH"))
    assert build_chirality_constraint(layout).terms == {}


# ---------------------------------------------------------------------------
# Penalty weights
# ---------------------------------------------------------------------------


def test_derived_penalties_satisfy_the_papers_bound() -> None:
    """Every per-pair weight satisfies lambda_1 > 6(j-i+1) lambda_2 + eps.

    This is the condition stated after Eq. SI-30, and it is what makes claiming a
    contact unprofitable unless the beads really are at distance one.
    """
    instance = _instance("HHPHPPHH")
    epsilon = hp_contact_energy(instance)
    penalties = derive_penalties(instance, epsilon)
    for i, j in contact_pairs(instance.n_beads):
        assert penalties.lambda_1[(i, j)] > (
            6.0 * (j - i + 1) * penalties.lambda_2[(i, j)] + epsilon(i, j)
        )


def test_penalties_scale_with_the_instance_not_a_constant() -> None:
    """A sequence with more possible contacts gets proportionately larger weights.

    A hardcoded penalty would not do this, so this test is what distinguishes a derived
    weight from a magic number that happens to work.
    """
    short = _instance("HHPHPPHH")
    small = derive_penalties(short, hp_contact_energy(short))
    long_chain = _instance("HHHHHHHHHH")
    large = derive_penalties(long_chain, hp_contact_energy(long_chain))
    assert large.energy_scale > small.energy_scale
    assert large.back > small.back


def test_penalty_below_the_bound_breaks_the_ground_state() -> None:
    """Weakening the derived penalty produces a ground state that is not a conformation.

    This is what proves the bound does real work. With the derived weights the ground
    state is a valid self-avoiding fold at the exhaustive optimum; with lambda_1 set
    below the paper's condition, the Hamiltonian prefers to claim contacts that do not
    exist, and reports an energy below the true optimum.
    """
    instance = _instance("HHHHHHHH")
    epsilon = hp_contact_energy(instance)
    true_optimum = exhaustive_search(instance).energy

    sound = build_hamiltonian(instance)
    assert sound.ground_state()[0] == pytest.approx(true_optimum)

    derived = derive_penalties(instance, epsilon)
    crippled = replace(derived, lambda_1=dict.fromkeys(derived.lambda_1, 0.01))
    broken = build_hamiltonian(instance, penalties=crippled)
    assert broken.ground_state()[0] < true_optimum


# ---------------------------------------------------------------------------
# Equivalence of the three representations
# ---------------------------------------------------------------------------


def test_pauli_operator_is_diagonal_and_reproduces_the_core() -> None:
    """The Pauli form's diagonal equals the core polynomial at every bitstring.

    N = 6 is used rather than the smaller sizes the brief suggests because contacts only
    become geometrically possible at a separation of five bonds, so at N <= 5 the
    interaction term is empty and the comparison would prove nothing about it.
    """
    hamiltonian = build_hamiltonian(_instance("HPHPPH"))
    operator = hamiltonian.to_pauli()
    matrix = np.asarray(operator.to_matrix())

    assert np.allclose(matrix, np.diag(np.diag(matrix))), "operator is not diagonal"

    size = hamiltonian.layout.n_primary_qubits
    for index in range(2**size):
        # Qiskit is little-endian: qubit i is bit i of the basis-state index.
        assignment = {i: (index >> i) & 1 for i in range(size)}
        assert matrix[index, index].real == pytest.approx(
            hamiltonian.core.evaluate(assignment)
        )


def test_qubo_and_ising_describe_the_same_model() -> None:
    """The Ising view is the QUBO under ``q = (1 - s) / 2``, at every assignment."""
    hamiltonian = build_hamiltonian(_instance("HPHPPH"))
    qubo = hamiltonian.to_qubo()
    ising = hamiltonian.to_ising()
    rng = np.random.default_rng(20260911)
    for _ in range(200):
        bits = rng.integers(0, 2, size=qubo.n_variables)
        spins = 1 - 2 * bits
        assert qubo.energy(bits) == pytest.approx(ising.energy(spins))


def test_quadratic_views_reproduce_the_core_after_minimising_auxiliaries() -> None:
    """QUBO minimised over its auxiliaries equals the exact core, at every assignment.

    This is the precise statement of what Rosenberg reduction guarantees, checked over
    the whole primary assignment space rather than only at the optimum. A reduction that
    agreed only at the ground state would still mislead every solver that walks the
    landscape.
    """
    hamiltonian = build_hamiltonian(_instance("HPHPPH"))
    qubo = hamiltonian.to_qubo()
    n_primary = hamiltonian.layout.n_primary_qubits
    n_auxiliary = hamiltonian.layout.n_auxiliary_qubits

    for primary in itertools.product((0, 1), repeat=n_primary):
        exact = hamiltonian.core.evaluate(dict(enumerate(primary)))
        best = min(
            qubo.energy(np.array(primary + auxiliary))
            for auxiliary in itertools.product((0, 1), repeat=n_auxiliary)
        )
        assert best == pytest.approx(exact)


# ---------------------------------------------------------------------------
# Agreement with the M1 exhaustive baseline
# ---------------------------------------------------------------------------

SEQUENCES = ("HPHPPH", "HHPPHH", "HHHHHH", "HHHHHHH", "HHPHPPHH", "HPHPPHHPH")


@pytest.mark.parametrize("sequence", SEQUENCES)
def test_strict_hamiltonian_ground_state_matches_the_exhaustive_optimum(
    sequence: str,
) -> None:
    """With global self-avoidance enforced, the ground state is M1's optimum exactly."""
    instance = _instance(sequence)
    hamiltonian = build_hamiltonian(instance, enforce_global_saw=True)
    energy, turns, _ = hamiltonian.ground_state()
    assert energy == pytest.approx(exhaustive_search(instance).energy)
    assert LATTICE.is_self_avoiding(turns)


@pytest.mark.parametrize("sequence", SEQUENCES)
def test_paper_faithful_ground_state_also_decodes_to_a_valid_conformation(
    sequence: str,
) -> None:
    """The published Hamiltonian's ground state is a self-avoiding fold at these sizes.

    The published model protects only against overlaps adjacent to a claimed contact,
    so global self-avoidance is not guaranteed by construction. Measured over every
    sequence tested here it nonetheless holds, because an overlap only pays if it buys
    an extra contact, and the neighbourhood of every claimed contact is exactly where
    the penalty applies.

    This is an empirical result at N <= 9, not a proof. That is why
    ``enforce_global_saw`` exists, and why RESULTS.md states the distinction.
    """
    instance = _instance(sequence)
    energy, turns, _ = build_hamiltonian(instance).ground_state()
    assert energy == pytest.approx(exhaustive_search(instance).energy)
    assert LATTICE.is_self_avoiding(turns)


def test_decoded_ground_state_contacts_are_real() -> None:
    """Every contact qubit set in the ground state corresponds to a genuine contact."""
    instance = _instance("HHPHPPHH")
    _, turns, contacts = build_hamiltonian(instance).ground_state()
    positions = LATTICE.walk(turns)
    for (i, j), claimed in contacts.items():
        actual = LATTICE.are_nearest_neighbours(positions[i], positions[j])
        if claimed:
            assert actual, f"claimed a contact that does not exist: {(i, j)}"


# ---------------------------------------------------------------------------
# Resource accounting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_beads", [6, 7, 8, 9, 10])
def test_configuration_register_is_two_per_free_turn(n_beads: int) -> None:
    """The turn register is ``2(N-3)``, the count reported by the reference paper."""
    layout = build_layout(_instance("H" * n_beads))
    assert layout.n_turn_qubits == 2 * (n_beads - 3)


def test_qubit_counts_are_reported_separately() -> None:
    """Turn, contact and auxiliary registers are distinguishable, never merged.

    Quoting ``2(N-3)`` as the width of the whole Hamiltonian would understate it badly:
    the contact register grows quadratically and the auxiliaries faster still.
    """
    hamiltonian = build_hamiltonian(_instance("HHPHPPHH"))
    layout = hamiltonian.layout
    assert layout.n_turn_qubits == 10
    assert layout.n_contact_qubits == len(contact_pairs(8))
    assert layout.n_auxiliary_qubits > 0
    assert layout.total_qubits == (
        layout.n_turn_qubits + layout.n_contact_qubits + layout.n_auxiliary_qubits
    )
    assert layout.total_qubits > layout.n_turn_qubits


@pytest.mark.parametrize("n_beads", [6, 7, 8, 9])
def test_published_model_locality_does_not_grow_with_chain_length(
    n_beads: int,
) -> None:
    """The paper-faithful Hamiltonian stays 5-local at every chain length.

    Locality independent of ``N`` is the central resource claim of the reference paper
    (its Table S2 contrasts this with models whose locality grows as ``N``), and the
    reason the dense encoding is worth its extra degree over the sparse one. Reproducing
    it from this construction is what makes the claim ours rather than quoted.
    """
    assert build_hamiltonian(_instance("H" * n_beads)).core_degree == 5


def test_strict_mode_costs_degree_and_auxiliaries() -> None:
    """Enforcing global self-avoidance raises both the degree and the variable count.

    The quantitative version of the trade-off: guaranteeing validity is affordable
    classically and not on a near-term device.
    """
    instance = _instance("HHHHHHHH")
    faithful = build_hamiltonian(instance)
    strict = build_hamiltonian(instance, enforce_global_saw=True)
    assert strict.core_degree > faithful.core_degree
    assert strict.layout.n_auxiliary_qubits > faithful.layout.n_auxiliary_qubits
    assert strict.layout.n_turn_qubits == faithful.layout.n_turn_qubits


def test_only_the_tetrahedral_lattice_is_accepted() -> None:
    """The turn encoding is tetrahedral-specific and refuses other geometries."""
    from foldq.lattice import CubicLattice

    instance = FoldingInstance(Peptide("HPHPPH"), CubicLattice())
    with pytest.raises(ValueError, match="tetrahedral"):
        build_hamiltonian(instance)


def test_turn_assignment_rejects_conformations_that_break_symmetry_fixing() -> None:
    """A turn sequence not starting with the fixed prefix cannot be encoded."""
    hamiltonian = build_hamiltonian(_instance("HPHPPH"))
    bad = (FIXED_TURNS[0], FIXED_TURNS[1] ^ 1, 0, 0, 0)
    with pytest.raises(ValueError, match="fixed"):
        hamiltonian.turn_assignment(bad)
