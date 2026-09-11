"""Peptide sequences, residue classes and the HP contact energy.

The HP model reduces the twenty amino acids to two classes, hydrophobic and polar, and
scores a conformation by counting non-bonded hydrophobic-hydrophobic contacts at unit
lattice distance. It is the starting point here because its energies are small
integers, which makes every intermediate result checkable by hand.

References
----------
Lau, K. F. and Dill, K. A. "A lattice statistical mechanics model of the conformational
and sequence spaces of proteins." Macromolecules 22, 3986-3997 (1989).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from foldq.lattice import Lattice

#: The two residue classes of the HP model.
HYDROPHOBIC = "H"
POLAR = "P"
_ALPHABET = frozenset({HYDROPHOBIC, POLAR})

#: Energy awarded to one non-bonded hydrophobic-hydrophobic contact. The scale is
#: arbitrary in the HP model -- only the ratio to other terms matters -- and -1 is the
#: convention used throughout the literature.
HH_CONTACT_ENERGY = -1


@dataclass(frozen=True)
class Peptide:
    """A peptide written in the HP alphabet.

    Parameters
    ----------
    sequence
        Residue classes in chain order, e.g. ``"HPPHPH"``. Case is normalised.
    """

    sequence: str

    def __post_init__(self) -> None:
        """Normalise the sequence to upper case and reject anything malformed."""
        normalised = self.sequence.upper()
        object.__setattr__(self, "sequence", normalised)

        if len(normalised) < 2:
            message = f"sequence must have at least two residues, got {normalised!r}"
            raise ValueError(message)
        unknown = sorted(set(normalised) - _ALPHABET)
        if unknown:
            message = (
                f"sequence contains residues outside the HP alphabet: {unknown}; "
                f"expected only {sorted(_ALPHABET)}"
            )
            raise ValueError(message)

    @property
    def length(self) -> int:
        """Number of residues in the chain."""
        return len(self.sequence)

    def is_hydrophobic(self, index: int) -> bool:
        """Return whether the residue at ``index`` is hydrophobic."""
        return self.sequence[index] == HYDROPHOBIC


def hp_energy(peptide: Peptide, contacts: Sequence[tuple[int, int]]) -> int:
    """Return the HP energy of a conformation given its non-bonded contact pairs.

    Only contacts between two hydrophobic residues are scored. Polar-polar and
    hydrophobic-polar contacts contribute nothing, which is what drives hydrophobic
    residues into a collapsed core in this model.

    Parameters
    ----------
    peptide
        The sequence being folded.
    contacts
        Index pairs of non-bonded beads at unit lattice distance, as returned by
        :meth:`foldq.lattice.Lattice.contact_pairs`.
    """
    return HH_CONTACT_ENERGY * sum(
        1
        for i, j in contacts
        if peptide.is_hydrophobic(i) and peptide.is_hydrophobic(j)
    )


@dataclass(frozen=True)
class FoldingInstance:
    """One problem instance: a sequence, a lattice, and nothing stochastic.

    Every solver in the benchmark is handed the *same* instance. That is what makes
    the comparison meaningful, so this type deliberately carries no solver settings and
    no random state.
    """

    peptide: Peptide
    lattice: Lattice

    @property
    def n_beads(self) -> int:
        """Number of beads in the chain."""
        return self.peptide.length

    @property
    def n_turns(self) -> int:
        """Number of bonds, which is one fewer than the number of beads."""
        return self.n_beads - 1

    def energy(self, turns: Sequence[int]) -> int:
        """Return the energy of a turn sequence, or raise if it is not self-avoiding.

        Callers that enumerate candidates should filter with
        :meth:`foldq.lattice.Lattice.is_self_avoiding` first; the exception is a guard
        against a conformation that violates excluded volume being scored as though it
        were physical.
        """
        positions = self.lattice.walk(turns)
        if len(set(positions)) != len(positions):
            message = f"turn sequence {tuple(turns)} is not self-avoiding"
            raise ValueError(message)
        return hp_energy(self.peptide, self.lattice.contact_pairs(positions))

    def enumerate_conformations(self) -> Iterator[tuple[int, ...]]:
        """Yield every self-avoiding turn sequence for this instance.

        No symmetry reduction is applied. This is the full conformation space, used for
        validation rather than for solving.
        """
        for turns in itertools.product(
            range(self.lattice.n_directions), repeat=self.n_turns
        ):
            if self.lattice.is_self_avoiding(turns):
                yield turns
