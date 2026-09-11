"""Turn encoding, qubit indexing and symmetry fixing for the tetrahedral lattice.

A conformation of an ``N``-bead chain is ``N - 1`` turns, each one of four directions,
so two qubits encode one turn with no wasted states. The first two turns are held at
fixed values, which removes four qubits and quotients out global rotation.

This module is specific to the tetrahedral lattice. The cubic lattice in
:mod:`foldq.lattice` has six directions, which do not fit a power of two, and exists
only as a validation geometry.

References
----------
Robert, A., Barkoutsos, P. K., Woerner, S. and Tavernelli, I.
"Resource-efficient quantum algorithm for protein folding."
npj Quantum Information 7, 38 (2021).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence

#: Number of directions at a tetrahedral lattice vertex.
N_DIRECTIONS = 4

#: Qubits needed to encode one turn: log2(4).
QUBITS_PER_TURN = 2

#: The canonical values held fixed for the first two turns.
#:
#: Derivation. The proper rotation group of a regular tetrahedron is isomorphic to A4,
#: of order 12, and acts on the four directions as the even permutations. Rotating a
#: conformation leaves its energy unchanged, so only one representative of each
#: rotational orbit needs to be searched.
#:
#: A4 is transitive on the four directions, so some rotation maps the first turn to any
#: chosen value -- here 0. The subgroup fixing that direction is the three-fold
#: rotation about it, which is transitive on the remaining three directions, so a
#: further rotation maps the second turn to any chosen value differing from the first
#: -- here 1. The second turn does differ from the first in every self-avoiding
#: conformation, because on this lattice a repeated turn index is a back-track.
#:
#: So exactly one conformation per rotational orbit has this prefix: the reduction
#: factor is the group order, 12, even though fixing two four-valued turns might
#: naively suggest 16. The saving is 2 turns x 2 qubits = 4 qubits, for free.
FIXED_TURNS: tuple[int, int] = (0, 1)

#: Number of turns held fixed by :data:`FIXED_TURNS`.
N_FIXED_TURNS = len(FIXED_TURNS)


def n_free_turns(n_beads: int) -> int:
    """Return the number of turns that remain free after symmetry fixing."""
    if n_beads < N_FIXED_TURNS + 1:
        message = (
            f"need at least {N_FIXED_TURNS + 1} beads to fix "
            f"{N_FIXED_TURNS} turns, got {n_beads}"
        )
        raise ValueError(message)
    return (n_beads - 1) - N_FIXED_TURNS


def n_qubits(n_beads: int) -> int:
    """Return the number of qubits needed to encode an ``n_beads``-long chain.

    Two qubits per free turn. For the 10-residue peptide the brief uses as its
    reference point: 9 turns, 2 fixed, 7 free, 14 qubits.

    Note that this counts the *turn* register only. The interaction term of the
    Hamiltonian introduced in M2 needs additional ancilla qubits to carry the pairwise
    contact indicators, so the total circuit width will exceed this number.
    """
    return QUBITS_PER_TURN * n_free_turns(n_beads)


def bits_from_turns(turns: Sequence[int]) -> str:
    """Encode a turn sequence as a bitstring, two bits per turn.

    Turn ``t`` occupies bits ``2t`` and ``2t + 1``, most significant first within the
    pair, so ``(0, 1, 2, 3)`` encodes as ``"00011011"``.
    """
    for turn in turns:
        if not 0 <= turn < N_DIRECTIONS:
            message = f"turn {turn} outside 0..{N_DIRECTIONS - 1}"
            raise ValueError(message)
    return "".join(format(turn, "02b") for turn in turns)


def turns_from_bits(bits: str, n_turns: int) -> tuple[int, ...]:
    """Decode a bitstring into a turn sequence.

    This is what gives a measured bitstring a physical meaning: each pair of bits names
    one of the four lattice directions, and walking those directions out reconstructs
    the conformation.
    """
    expected = QUBITS_PER_TURN * n_turns
    if len(bits) != expected:
        message = f"expected {expected} bits for {n_turns} turns, got {len(bits)}"
        raise ValueError(message)
    return tuple(
        int(bits[i : i + QUBITS_PER_TURN], 2)
        for i in range(0, expected, QUBITS_PER_TURN)
    )


def enumerate_turns(
    n_turns: int, *, fix_symmetry: bool = True
) -> Iterator[tuple[int, ...]]:
    """Yield candidate turn sequences, optionally restricted to the reduced space.

    Parameters
    ----------
    n_turns
        Number of turns in the chain, one fewer than the number of beads.
    fix_symmetry
        When true, hold the first two turns at :data:`FIXED_TURNS`, yielding one
        representative per rotational orbit. When false, yield all ``4**n_turns``
        sequences, which is what the reduced search is validated against.

    Sequences are yielded whether or not they are self-avoiding; filtering is the
    caller's job, because the exhaustive solver needs to count candidates examined.
    """
    if n_turns < 0:
        message = f"n_turns must be non-negative, got {n_turns}"
        raise ValueError(message)

    if not fix_symmetry:
        yield from itertools.product(range(N_DIRECTIONS), repeat=n_turns)
        return

    if n_turns < N_FIXED_TURNS:
        message = (
            f"cannot fix {N_FIXED_TURNS} turns in a chain with {n_turns} turns; "
            "pass fix_symmetry=False for chains this short"
        )
        raise ValueError(message)

    for tail in itertools.product(range(N_DIRECTIONS), repeat=n_turns - N_FIXED_TURNS):
        yield (*FIXED_TURNS, *tail)
