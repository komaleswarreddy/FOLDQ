"""Geometry tests for the lattice models.

The tetrahedral (diamond) lattice is the one the qubit encoding is built on; the
simple cubic lattice exists only so that the geometry and self-avoidance code can be
validated against independently published walk counts.

Sources
-------
Cubic self-avoiding walk counts: OEIS A001412, "Number of n-step self-avoiding walks
on cubic lattice".  https://oeis.org/A001412

Diamond self-avoiding walk counts: reconstructed from OEIS A227715 and A227716, which
give n-step diamond-lattice walks resolved by the x-coordinate of the endpoint.  See
``test_diamond_walk_counts_match_oeis`` for the reconstruction and its own check.
"""

from __future__ import annotations

import pytest

from foldq.lattice import CubicLattice, Lattice, TetrahedralLattice

# ---------------------------------------------------------------------------
# Hand-computed positions
# ---------------------------------------------------------------------------


def test_tetrahedral_positions_for_four_beads_are_hand_computable() -> None:
    """Turns (0, 1, 2) give positions that can be worked out on paper.

    The diamond lattice is two interpenetrating sublattices, so the step taken by bead
    ``i`` is ``(-1)**i`` times the direction vector. Working it out by hand::

        p0 = (0, 0, 0)
        p1 = p0 + d0 = (0,0,0) + ( 1,  1,  1) = ( 1, 1, 1)
        p2 = p1 - d1 = (1,1,1) - ( 1, -1, -1) = ( 0, 2, 2)
        p3 = p2 + d2 = (0,2,2) + (-1,  1, -1) = (-1, 3, 1)
    """
    lattice = TetrahedralLattice()
    assert lattice.walk((0, 1, 2)) == [
        (0, 0, 0),
        (1, 1, 1),
        (0, 2, 2),
        (-1, 3, 1),
    ]


def test_cubic_positions_for_four_beads_are_hand_computable() -> None:
    """Turns (+x, +y, +z) walk one unit along each axis in turn."""
    lattice = CubicLattice()
    assert lattice.walk((0, 2, 4)) == [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
    ]


@pytest.mark.parametrize("lattice", [TetrahedralLattice(), CubicLattice()])
def test_every_bond_has_unit_lattice_length(lattice: Lattice) -> None:
    """Consecutive beads sit at one nearest-neighbour separation, for every turn."""
    for turn in range(lattice.n_directions):
        positions = lattice.walk((turn,))
        assert lattice.squared_distance(positions[0], positions[1]) == (
            lattice.nearest_neighbour_squared_distance
        )


def test_tetrahedral_step_direction_alternates_with_bead_parity() -> None:
    """Odd-indexed beads step along the negated direction vector.

    This is the property that distinguishes a diamond lattice from a naive "four fixed
    directions" model. Getting it wrong still produces plausible-looking coordinates,
    so it is asserted directly rather than only through its consequences.
    """
    lattice = TetrahedralLattice()
    for turn in range(lattice.n_directions):
        even_step = lattice.step(0, turn)
        odd_step = lattice.step(1, turn)
        assert odd_step == tuple(-component for component in even_step)


# ---------------------------------------------------------------------------
# Self-avoidance
# ---------------------------------------------------------------------------


def test_tetrahedral_backtracking_is_a_repeated_turn_index() -> None:
    """Repeating a turn index returns to the previous-but-one bead.

    Because the step alternates sign, ``p_{i+2} - p_i = (-1)**i (d_a - d_b)`` for turns
    ``a`` then ``b``, which vanishes exactly when ``a == b``. So on this lattice a
    back-track is a *repeated* index, not an opposite one -- there is no "opposite"
    direction among the four.
    """
    lattice = TetrahedralLattice()
    for turn in range(lattice.n_directions):
        positions = lattice.walk((turn, turn))
        assert positions[2] == positions[0]
        assert not lattice.is_self_avoiding((turn, turn))


def test_tetrahedral_six_ring_is_rejected() -> None:
    """The turn sequence (0,1,2,0,1,2) closes the smallest diamond-lattice ring.

    Its displacement telescopes to zero::

        d0 - d1 + d2 - d0 + d1 - d2 = 0

    so bead 6 lands exactly on bead 0. The first five beads are distinct, which makes
    this a test of genuine self-intersection rather than of back-tracking.
    """
    lattice = TetrahedralLattice()
    positions = lattice.walk((0, 1, 2, 0, 1, 2))
    assert positions[6] == positions[0]
    assert len(set(positions[:6])) == 6
    assert not lattice.is_self_avoiding((0, 1, 2, 0, 1, 2))
    assert lattice.is_self_avoiding((0, 1, 2, 0, 1))


def test_cubic_square_ring_is_rejected() -> None:
    """+x, +y, -x, -y closes a unit square, the smallest cubic-lattice ring."""
    lattice = CubicLattice()
    assert not lattice.is_self_avoiding((0, 2, 1, 3))
    assert lattice.is_self_avoiding((0, 2, 1))


# ---------------------------------------------------------------------------
# External validation: published self-avoiding walk counts
# ---------------------------------------------------------------------------

#: OEIS A001412, n-step self-avoiding walks on the simple cubic lattice, n = 0..9.
CUBIC_SAW_COUNTS = (1, 6, 30, 150, 726, 3534, 16926, 81390, 387966, 1853886)

#: Total n-step self-avoiding walks on the diamond lattice, n = 0..9, reconstructed
#: from the OEIS triangles A227715 (even n) and A227716 (odd n). Those tabulate walks
#: by the x-coordinate of the endpoint for x >= 0; the x < 0 half follows by mirror
#: symmetry, so the total is ``row[0] + 2 * sum(row[1:])`` for even n (x = 0 is its own
#: mirror) and ``2 * sum(row)`` for odd n (no endpoint has x = 0).
DIAMOND_SAW_COUNTS = (1, 4, 12, 36, 108, 324, 948, 2796, 8196, 24060)


def test_diamond_saw_reconstruction_matches_the_analytic_regime() -> None:
    """The reconstructed diamond counts agree with theory where theory is exact.

    No self-intersection is possible on the diamond lattice below six steps, because
    its smallest ring has six bonds. In that regime the count is exactly the number of
    non-back-tracking walks, ``4 * 3**(n-1)``. Agreement there is what justifies
    trusting the same reconstruction at n >= 6, where the answer is not obvious.
    """
    for steps in range(1, 6):
        assert DIAMOND_SAW_COUNTS[steps] == 4 * 3 ** (steps - 1)
    # The first value where self-avoidance actually bites.
    assert DIAMOND_SAW_COUNTS[6] < 4 * 3**5


@pytest.mark.parametrize(
    ("lattice", "expected"),
    [
        # The nine-step cubic count is enumerated in the slow test below; counting it
        # here would add about five seconds to every run of the suite.
        (CubicLattice(), CUBIC_SAW_COUNTS[:9]),
        (TetrahedralLattice(), DIAMOND_SAW_COUNTS),
    ],
    ids=["cubic", "tetrahedral"],
)
def test_walk_counts_match_oeis(lattice: Lattice, expected: tuple[int, ...]) -> None:
    """Counting self-avoiding walks reproduces the published totals.

    This validates the direction vectors, the parity handling and the self-avoidance
    predicate together against numbers nobody in this repository chose.
    """
    for steps, count in enumerate(expected):
        assert lattice.count_self_avoiding_walks(steps) == count, f"{steps=}"


@pytest.mark.slow
def test_nine_step_cubic_walk_count_matches_oeis() -> None:
    """The largest cubic count checked here, 1 853 886 nine-step walks (A001412)."""
    assert CubicLattice().count_self_avoiding_walks(9) == CUBIC_SAW_COUNTS[9]
