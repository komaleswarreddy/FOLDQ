"""Lattice geometries for coarse-grained chain conformations.

Two lattices are provided:

``TetrahedralLattice``
    The diamond lattice. Every vertex has exactly four neighbours, which maps onto two
    qubits per turn with no wasted states, so this is the lattice the qubit encoding is
    built on. Chosen over a cubic lattice for that reason: six directions would need
    three qubits per turn and waste two of the eight states, and those wasted states
    would need penalty terms of their own.

``CubicLattice``
    The simple cubic lattice. Present only so that the walk-generation and
    self-avoidance code can be checked against independently published enumeration
    results (OEIS A001412); nothing in the qubit encoding uses it.

References
----------
Robert, A., Barkoutsos, P. K., Woerner, S. and Tavernelli, I.
"Resource-efficient quantum algorithm for protein folding."
npj Quantum Information 7, 38 (2021). Source of the tetrahedral turn encoding used here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

#: An integer lattice position. Integer arithmetic keeps positions exact and hashable,
#: which matters because self-avoidance is decided by set membership.
Coord = tuple[int, int, int]


class Lattice(ABC):
    """A lattice on which a chain of beads is embedded.

    Subclasses supply a set of direction vectors and a rule mapping a bead index and a
    turn index to a displacement. Everything else -- walking a turn sequence out into
    coordinates, deciding self-avoidance, finding contacts -- is shared.
    """

    #: Human-readable name, used in artifacts and figures.
    name: ClassVar[str]

    #: Direction vectors, indexed by turn.
    directions: ClassVar[tuple[Coord, ...]]

    #: Squared distance between two vertices that are nearest neighbours. Squared, so
    #: the comparison stays in exact integer arithmetic.
    nearest_neighbour_squared_distance: ClassVar[int]

    @property
    def n_directions(self) -> int:
        """Number of turns available at each bead."""
        return len(self.directions)

    @abstractmethod
    def step(self, bead_index: int, turn: int) -> Coord:
        """Return the displacement taken by the bond leaving bead ``bead_index``.

        Parameters
        ----------
        bead_index
            Index of the bead the bond starts from. Some lattices, notably the diamond
            lattice, take a displacement that depends on this index.
        turn
            Which of the available directions to take.
        """

    @staticmethod
    def squared_distance(left: Coord, right: Coord) -> int:
        """Return the squared Euclidean distance between two lattice positions."""
        return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))

    def are_nearest_neighbours(self, left: Coord, right: Coord) -> bool:
        """Return whether two positions are one lattice bond apart."""
        return (
            self.squared_distance(left, right)
            == self.nearest_neighbour_squared_distance
        )

    def walk(self, turns: Sequence[int]) -> list[Coord]:
        """Walk a turn sequence out into bead positions, starting at the origin.

        A chain of ``N`` beads has ``N - 1`` turns, so the returned list is one longer
        than ``turns``. The first bead is pinned at the origin, which quotients out
        translational symmetry at zero cost.
        """
        positions: list[Coord] = [(0, 0, 0)]
        for index, turn in enumerate(turns):
            if not 0 <= turn < self.n_directions:
                message = f"turn {turn} out of range for {self.name} lattice"
                raise ValueError(message)
            dx, dy, dz = self.step(index, turn)
            x, y, z = positions[-1]
            positions.append((x + dx, y + dy, z + dz))
        return positions

    def is_self_avoiding(self, turns: Sequence[int]) -> bool:
        """Return whether a turn sequence places every bead on a distinct vertex.

        This is the excluded-volume constraint: two beads cannot occupy the same point
        in space. It subsumes back-tracking, which is just the shortest way to violate
        it.
        """
        positions = self.walk(turns)
        return len(set(positions)) == len(positions)

    def count_self_avoiding_walks(self, steps: int) -> int:
        """Count the self-avoiding walks of a given number of steps from the origin.

        Walks are counted as distinct step sequences, not up to lattice symmetry,
        matching the convention of the published enumerations this is validated
        against.

        The search prunes as soon as a partial walk revisits a vertex, so the cost is
        proportional to the number of self-avoiding walks rather than to the
        exponentially larger number of unconstrained ones.
        """
        if steps < 0:
            message = f"steps must be non-negative, got {steps}"
            raise ValueError(message)

        origin: Coord = (0, 0, 0)

        def extend(index: int, position: Coord, visited: set[Coord]) -> int:
            if index == steps:
                return 1
            total = 0
            for turn in range(self.n_directions):
                dx, dy, dz = self.step(index, turn)
                nxt = (position[0] + dx, position[1] + dy, position[2] + dz)
                if nxt in visited:
                    continue
                visited.add(nxt)
                total += extend(index + 1, nxt, visited)
                visited.remove(nxt)
            return total

        return extend(0, origin, {origin})

    def contact_pairs(self, positions: Sequence[Coord]) -> list[tuple[int, int]]:
        """Return index pairs of non-bonded beads that are nearest neighbours.

        Beads separated by one bond are excluded because they are bonded, not in
        contact. Beads separated by two bonds are excluded as well, and on both
        lattices here they cannot be nearest neighbours anyway: on the cubic lattice
        they are at squared distance 0 or 4, and on the diamond lattice they sit on the
        same sublattice, which contains no nearest-neighbour pairs.
        """
        pairs: list[tuple[int, int]] = []
        for i in range(len(positions)):
            for j in range(i + 3, len(positions)):
                if self.are_nearest_neighbours(positions[i], positions[j]):
                    pairs.append((i, j))
        return pairs


class TetrahedralLattice(Lattice):
    """The diamond lattice, with four directions per vertex.

    The diamond lattice is two interpenetrating face-centred-cubic sublattices. A bead
    on one sublattice steps along ``+d``; a bead on the other steps along ``-d``. With
    beads numbered from zero along the chain, the sublattice is just the bead index
    parity, so the bond leaving bead ``i`` is ``(-1)**i * d[turn]``.

    Ignoring that alternation is a subtle error: it produces coordinates that look
    reasonable and bond lengths that are all correct, but the wrong lattice.

    Two consequences are relied on elsewhere:

    * A back-track is a *repeated* turn index. ``p[i+2] - p[i] = (-1)**i (d_a - d_b)``
      for consecutive turns ``a`` then ``b``, which is zero exactly when ``a == b``.
      There is no "opposite" direction among the four.
    * Beads whose indices differ by an even number sit on the same sublattice and can
      never be in contact, because every nearest-neighbour bond joins the sublattices.
    """

    name: ClassVar[str] = "tetrahedral"

    #: The four vectors from a vertex to its neighbours, at the tetrahedral angle
    #: (their pairwise dot product is -1/3 of their squared length) and summing to
    #: zero. Written unnormalised so coordinates stay integral; the physical bond
    #: length is sqrt(3) in these units.
    directions: ClassVar[tuple[Coord, ...]] = (
        (1, 1, 1),
        (1, -1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
    )

    nearest_neighbour_squared_distance: ClassVar[int] = 3

    def step(self, bead_index: int, turn: int) -> Coord:
        """Return ``(-1)**bead_index`` times the chosen direction vector."""
        dx, dy, dz = self.directions[turn]
        if bead_index % 2 == 0:
            return (dx, dy, dz)
        return (-dx, -dy, -dz)


class CubicLattice(Lattice):
    """The simple cubic lattice, with six axis-aligned directions per vertex.

    Validation geometry only. It is here because the number of self-avoiding walks on
    the cubic lattice is published (OEIS A001412), which gives the walk generator and
    the self-avoidance predicate an external check at sizes small enough to enumerate
    exhaustively. No qubit encoding uses it: six directions do not fit a power of two.
    """

    name: ClassVar[str] = "cubic"

    directions: ClassVar[tuple[Coord, ...]] = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )

    nearest_neighbour_squared_distance: ClassVar[int] = 1

    def step(self, bead_index: int, turn: int) -> Coord:  # noqa: ARG002
        """Return the chosen direction vector, which does not depend on the bead.

        ``bead_index`` is unused: unlike the diamond lattice, the cubic lattice is a
        single sublattice, so every vertex offers the same six displacements. The
        parameter is kept to satisfy the :class:`Lattice` interface.
        """
        return self.directions[turn]
