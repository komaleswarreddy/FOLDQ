"""Tests for sequence handling and the HP contact energy.

The HP model assigns every residue to one of two classes and scores a conformation by
counting non-bonded hydrophobic-hydrophobic contacts at unit lattice distance, one
unit of energy each.

References
----------
Lau, K. F. and Dill, K. A. "A lattice statistical mechanics model of the conformational
and sequence spaces of proteins." Macromolecules 22, 3986-3997 (1989). Origin of the
HP model and of the convention that only non-bonded H-H contacts are scored.
"""

from __future__ import annotations

import pytest

from foldq.lattice import CubicLattice, TetrahedralLattice
from foldq.peptide import FoldingInstance, Peptide, hp_energy


def test_sequence_is_parsed_into_residue_classes() -> None:
    """An HP string exposes its length and per-residue hydrophobicity."""
    peptide = Peptide("HPPH")
    assert peptide.length == 4
    assert [peptide.is_hydrophobic(i) for i in range(4)] == [True, False, False, True]


def test_lowercase_sequences_are_accepted() -> None:
    """Case is normalised, because published sequences appear in both forms."""
    assert Peptide("hphp").sequence == "HPHP"


@pytest.mark.parametrize("bad", ["", "H", "HPX", "HP-H", "123"])
def test_invalid_sequences_are_rejected(bad: str) -> None:
    """Anything that is not at least two residues drawn from {H, P} is an error."""
    with pytest.raises(ValueError, match="sequence"):
        Peptide(bad)


def test_energy_counts_only_non_bonded_hydrophobic_contacts() -> None:
    """A contact scores -1 only when both partners are hydrophobic.

    Worked by hand on the cubic lattice. The turn sequence +x, +y, -x closes three
    sides of a unit square, so beads 0 and 3 end up one lattice unit apart and are the
    only non-bonded pair in contact::

        p0 = (0,0,0)  p1 = (1,0,0)  p2 = (1,1,0)  p3 = (0,1,0)

    With sequence HPPH that pair is H-H and the energy is -1. With HPPP it is H-P and
    the energy is 0.
    """
    lattice = CubicLattice()
    positions = lattice.walk((0, 2, 1))
    pairs = lattice.contact_pairs(positions)
    assert pairs == [(0, 3)]
    assert hp_energy(Peptide("HPPH"), pairs) == -1
    assert hp_energy(Peptide("HPPP"), pairs) == 0
    assert hp_energy(Peptide("PPPH"), pairs) == 0


def test_bonded_neighbours_are_never_counted_as_contacts() -> None:
    """Beads one or two bonds apart are excluded from the contact list.

    Consecutive beads are always at unit distance by construction, so counting them
    would add a constant to every conformation's energy and score nothing physical.
    """
    lattice = CubicLattice()
    positions = lattice.walk((0, 2, 4))
    assert lattice.contact_pairs(positions) == []
    assert hp_energy(Peptide("HHHH"), lattice.contact_pairs(positions)) == 0


def test_tetrahedral_contacts_only_occur_at_odd_index_separation() -> None:
    """On the diamond lattice, contacting beads differ by an odd number of bonds.

    Every nearest-neighbour bond joins the two sublattices, and the sublattice of a
    bead is its index parity. So an even index separation cannot be a contact. This is
    a structural property of the lattice, and checking it over an exhaustive sweep
    guards against a direction-vector or parity error that the walk-count test might
    not localise.
    """
    lattice = TetrahedralLattice()
    instance = FoldingInstance(Peptide("HHHHHHHH"), lattice)
    seen = 0
    for turns in instance.enumerate_conformations():
        for i, j in lattice.contact_pairs(lattice.walk(turns)):
            assert (j - i) % 2 == 1
            seen += 1
    assert seen > 0, "no contacts found at all, so the assertion proved nothing"


def test_instance_rejects_a_sequence_that_does_not_match_its_length() -> None:
    """A folding instance ties one sequence to one chain length."""
    instance = FoldingInstance(Peptide("HPHP"), TetrahedralLattice())
    assert instance.n_beads == 4
    assert instance.n_turns == 3
