"""Tests for the turn encoding, qubit indexing and symmetry fixing.

The encoding maps a conformation to a bitstring: two qubits per turn on the
tetrahedral lattice, with the first two turns held at fixed values to quotient out
global rotation.

References
----------
Robert, A., Barkoutsos, P. K., Woerner, S. and Tavernelli, I.
"Resource-efficient quantum algorithm for protein folding."
npj Quantum Information 7, 38 (2021).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable

import pytest

from foldq.encoding import (
    FIXED_TURNS,
    bits_from_turns,
    enumerate_turns,
    n_qubits,
    turns_from_bits,
)
from foldq.lattice import TetrahedralLattice
from foldq.peptide import FoldingInstance, Peptide, hp_energy


def test_two_qubits_per_free_turn() -> None:
    """A chain of N beads needs 2(N-1-2) qubits once two turns are fixed.

    Worked through for the 10-residue peptide the brief asks about: 9 turns, 2 of them
    fixed, 7 free turns, 14 qubits.
    """
    assert n_qubits(10) == 14
    assert n_qubits(4) == 2
    assert n_qubits(3) == 0


def test_chains_too_short_to_have_a_free_turn_are_rejected() -> None:
    """Fewer than three turns leaves nothing to encode after symmetry fixing."""
    with pytest.raises(ValueError, match="beads"):
        n_qubits(2)


@pytest.mark.parametrize("turns", [(0, 1, 2), (0, 1, 3, 2, 0), (0, 1, 0, 3)])
def test_bitstring_round_trip(turns: tuple[int, ...]) -> None:
    """Encoding a turn sequence and decoding it returns the original."""
    bits = bits_from_turns(turns)
    assert turns_from_bits(bits, n_turns=len(turns)) == turns


def test_bitstring_is_two_bits_per_turn_in_order() -> None:
    """The bit layout is explicit, not incidental.

    Turn ``t`` occupies bits ``2t`` and ``2t+1``, most significant first within the
    pair, so the sequence (0, 1, 2, 3) encodes as 00 01 10 11. Pinning this down is
    what makes a measured bitstring physically interpretable.
    """
    assert bits_from_turns((0, 1, 2, 3)) == "00011011"


def test_fixed_turns_are_distinct() -> None:
    """The two fixed turns differ, so the fixed prefix is not itself a back-track."""
    assert FIXED_TURNS[0] != FIXED_TURNS[1]


def test_enumeration_with_symmetry_fixing_holds_the_first_two_turns() -> None:
    """Every enumerated sequence starts with the canonical prefix."""
    for turns in enumerate_turns(n_turns=5, fix_symmetry=True):
        assert turns[:2] == FIXED_TURNS


def test_symmetry_fixing_reduces_the_search_space_by_the_rotation_group_order() -> None:
    """Fixing two turns keeps exactly one conformation per rotational orbit.

    The proper rotation group of the tetrahedron is isomorphic to A4, of order 12, and
    acts on the four directions as the even permutations. A4 is transitive on the four
    directions, so the first turn can always be rotated to a chosen value; the
    stabiliser of a direction is the three-fold rotation about it, which is transitive
    on the remaining three directions, so the second turn can also be fixed provided it
    differs from the first -- which it does for any self-avoiding conformation, since a
    repeated turn index is a back-track.

    Hence exactly 1 in 12 self-avoiding conformations survives, and the count ratio is
    the group order rather than the naive 16 that fixing two 4-valued turns suggests.
    """
    lattice = TetrahedralLattice()
    for n_turns in range(2, 7):
        full = [
            turns
            for turns in itertools.product(range(4), repeat=n_turns)
            if lattice.is_self_avoiding(turns)
        ]
        reduced = [
            turns
            for turns in enumerate_turns(n_turns=n_turns, fix_symmetry=True)
            if lattice.is_self_avoiding(turns)
        ]
        assert len(full) == 12 * len(reduced), f"{n_turns=}"


def test_symmetry_fixing_does_not_change_the_optimum() -> None:
    """The reduced search finds the same minimum energy as the full search.

    This is the criterion that matters: a symmetry reduction that lowers the qubit
    count but discards the ground state is worse than no reduction at all.
    """
    lattice = TetrahedralLattice()

    def best(peptide: Peptide, candidates: Iterable[tuple[int, ...]]) -> int:
        return min(
            hp_energy(peptide, lattice.contact_pairs(lattice.walk(turns)))
            for turns in candidates
            if lattice.is_self_avoiding(turns)
        )

    for sequence in ("HPHPPH", "HHPPHP", "HPPHPH", "HHPHPPH"):
        peptide = Peptide(sequence)
        n_turns = peptide.length - 1
        full = best(peptide, itertools.product(range(4), repeat=n_turns))
        reduced = best(peptide, enumerate_turns(n_turns=n_turns, fix_symmetry=True))
        assert full == reduced, f"{sequence=}"


def test_unreduced_enumeration_covers_the_whole_space() -> None:
    """Without symmetry fixing, enumeration yields all 4**n_turns sequences."""
    assert len(list(enumerate_turns(n_turns=4, fix_symmetry=False))) == 4**4


def test_qubit_count_matches_the_free_turns_of_an_instance() -> None:
    """The instance and the encoding agree on how many qubits a chain needs."""
    instance = FoldingInstance(Peptide("HPHPPHPHPP"), TetrahedralLattice())
    assert instance.n_beads == 10
    assert n_qubits(instance.n_beads) == 14
