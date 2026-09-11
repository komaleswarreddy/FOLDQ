"""Construction of the folding Hamiltonian and its three equivalent representations.

The Hamiltonian follows Robert et al. (2021) with the dense two-qubit-per-turn encoding.
One pseudo-Boolean polynomial is the single source of truth; the Pauli, QUBO and Ising
views are all exported from it, so they cannot disagree by construction.

Structure
---------
``H = H_gc + H_ch + H_in`` (Eq. 1 of the paper), with an optional fourth term.

``H_gc``
    Growth constraint. Penalises a turn that repeats the previous turn index, which on
    this lattice is exactly a back-track (Eq. SI-16, SI-17).

``H_ch``
    Chirality. Empty for a main-chain-only model: the constraint of Eq. SI-26 acts on
    side-chain beads, and this model has none. The hook is kept so the term can be
    filled in, rather than pretending it is implemented.

``H_in``
    Interaction. One contact qubit per candidate pair, gated so that the favourable
    contact energy is only collected when the beads really are at distance one
    (Eq. SI-29, SI-30).

``H_saw`` (optional, not in the paper)
    Global self-avoidance. See :func:`build_hamiltonian` for why the reference model
    does not include it and what it costs to add.

Distance
--------
The paper measures separation with a turn-counting metric rather than a Euclidean one.
With ``Delta n_a(i,j) = sum_{k=i}^{j-1} (-1)^k f_a(k)`` (Eq. SI-11) and
``d(i,j) = sum_a Delta n_a(i,j)**2``, the map to Euclidean distance is bijective
(Eq. SI-14). In this package's integer coordinates, where a bond has squared length 3,
that relationship is::

    squared_euclidean(i, j) = 4 * d(i, j) - (1 if (j - i) is odd else 0)

so ``d == 1`` is a nearest-neighbour contact and ``d == 0`` is an overlap. This identity
is verified exhaustively in the tests against the geometry in :mod:`foldq.lattice`.

References
----------
Robert, A., Barkoutsos, P. K., Woerner, S. and Tavernelli, I.
"Resource-efficient quantum algorithm for protein folding."
npj Quantum Information 7, 38 (2021); arXiv:1908.02163. Equation numbers of the form
SI-n refer to its Supporting Information.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from foldq.encoding import (
    FIXED_TURNS,
    N_DIRECTIONS,
    N_FIXED_TURNS,
    enumerate_turns,
)
from foldq.lattice import TetrahedralLattice
from foldq.peptide import FoldingInstance
from foldq.polynomial import (
    BinaryPolynomial,
    QuadratizationResult,
    quadratize,
    sum_polynomials,
)

if TYPE_CHECKING:
    from qiskit.quantum_info import SparsePauliOp

#: Value of ``d(i, j)`` at a first-nearest-neighbour contact.
CONTACT_DISTANCE = 1

#: Minimum separation along the chain at which a 1-NN contact is geometrically
#: possible. The paper states that beads fewer than five bonds apart cannot be nearest
#: neighbours (SI, Section C); the tests verify it by exhaustive enumeration rather than
#: taking it on trust.
MIN_CONTACT_SEPARATION = 5

#: A contact energy model: given two bead indices, return the energy awarded when they
#: are in contact. Negative is favourable.
ContactEnergy = Callable[[int, int], float]


def hp_contact_energy(instance: FoldingInstance) -> ContactEnergy:
    """Return the HP contact energy function for an instance.

    Every non-bonded hydrophobic-hydrophobic contact is worth -1; everything else is
    worth nothing.
    """

    def epsilon(i: int, j: int) -> float:
        both_hydrophobic = instance.peptide.is_hydrophobic(
            i
        ) and instance.peptide.is_hydrophobic(j)
        return -1.0 if both_hydrophobic else 0.0

    return epsilon


@dataclass(frozen=True)
class QubitLayout:
    """Where each variable lives, and how many there are of each kind.

    The three counts are reported separately and never silently merged. The
    configuration register alone is ``2(N-3)``, which is the number the reference paper
    plots; quoting it as the cost of the whole Hamiltonian would understate the width by
    the contact and auxiliary registers, which scale faster.
    """

    n_beads: int
    turn_qubits: Mapping[tuple[int, int], int]
    contact_qubits: Mapping[tuple[int, int], int]
    n_auxiliary_qubits: int = 0

    @property
    def n_turn_qubits(self) -> int:
        """Qubits encoding the conformation: ``2(N-3)`` after symmetry fixing."""
        return len(self.turn_qubits)

    @property
    def n_contact_qubits(self) -> int:
        """Ancillas carrying the pairwise contact indicators."""
        return len(self.contact_qubits)

    @property
    def n_primary_qubits(self) -> int:
        """Turn plus contact qubits: everything with a direct physical meaning."""
        return self.n_turn_qubits + self.n_contact_qubits

    @property
    def total_qubits(self) -> int:
        """Every variable, including quadratization auxiliaries."""
        return self.n_primary_qubits + self.n_auxiliary_qubits

    def with_auxiliaries(self, count: int) -> QubitLayout:
        """Return a copy recording how many auxiliaries quadratization introduced."""
        return QubitLayout(
            n_beads=self.n_beads,
            turn_qubits=self.turn_qubits,
            contact_qubits=self.contact_qubits,
            n_auxiliary_qubits=count,
        )


def build_layout(instance: FoldingInstance) -> QubitLayout:
    """Assign variable indices to turn qubits and contact ancillas.

    Turns 0 and 1 are fixed by symmetry (see :data:`foldq.encoding.FIXED_TURNS`) and get
    no qubits. Every later turn gets two, high bit first.

    Contact ancillas are created only for pairs that can actually touch: separation odd,
    because every nearest-neighbour bond joins the two sublattices, and at least
    :data:`MIN_CONTACT_SEPARATION`. Creating ancillas for the other pairs would inflate
    the reported qubit count with variables that are provably always zero.
    """
    turn_qubits: dict[tuple[int, int], int] = {}
    index = 0
    for turn in range(N_FIXED_TURNS, instance.n_turns):
        for bit in range(2):
            turn_qubits[(turn, bit)] = index
            index += 1

    contact_qubits: dict[tuple[int, int], int] = {}
    for i, j in contact_pairs(instance.n_beads):
        contact_qubits[(i, j)] = index
        index += 1

    return QubitLayout(
        n_beads=instance.n_beads,
        turn_qubits=turn_qubits,
        contact_qubits=contact_qubits,
    )


def contact_pairs(n_beads: int) -> list[tuple[int, int]]:
    """Return bead pairs that can form a first-nearest-neighbour contact."""
    return [
        (i, j)
        for i in range(n_beads)
        for j in range(i + MIN_CONTACT_SEPARATION, n_beads)
        if (j - i) % 2 == 1
    ]


def chain_neighbours(index: int, n_beads: int) -> list[int]:
    """Return the beads bonded to ``index`` along the main chain.

    This is ``N(i)`` in Eq. SI-30. With no side chains it is at most two beads, which is
    within the ``|N(i)| <= 3`` the paper's penalty bound assumes.
    """
    return [k for k in (index - 1, index + 1) if 0 <= k < n_beads]


def turn_indicator(layout: QubitLayout, turn: int, direction: int) -> BinaryPolynomial:
    """Return ``f_a(turn)``: 1 when ``turn`` points along ``direction``, else 0.

    For the dense encoding the paper gives (Eq. SI-3 to SI-6), with ``q1`` the high bit
    and ``q2`` the low bit of the turn::

        f_0 = (1 - q1)(1 - q2)
        f_1 = q2 (q2 - q1)   ==  q2 (1 - q1)
        f_2 = q1 (q1 - q2)   ==  q1 (1 - q2)
        f_3 = q1 q2

    written here in the second, reduced form, which is identical because ``q**2 == q``.
    The turn index is ``2*q1 + q2``.

    Turns held fixed by symmetry have no qubits, so their indicators are constants.
    """
    if turn < N_FIXED_TURNS:
        return BinaryPolynomial.constant(1.0 if FIXED_TURNS[turn] == direction else 0.0)

    q1 = BinaryPolynomial.variable(layout.turn_qubits[(turn, 0)])
    q2 = BinaryPolynomial.variable(layout.turn_qubits[(turn, 1)])
    if direction == 0:
        return (1 - q1) * (1 - q2)
    if direction == 1:
        return q2 * (1 - q1)
    if direction == 2:
        return q1 * (1 - q2)
    if direction == 3:
        return q1 * q2
    message = f"direction {direction} outside 0..{N_DIRECTIONS - 1}"
    raise ValueError(message)


def delta_n(layout: QubitLayout, i: int, j: int, direction: int) -> BinaryPolynomial:
    """Return ``Delta n_a(i, j)`` from Eq. SI-11.

    The signed count of turns along ``direction`` between beads ``i`` and ``j``. The
    alternating sign is the sublattice alternation of the diamond lattice, the same
    ``(-1)**k`` that appears in :meth:`foldq.lattice.TetrahedralLattice.step`.
    """
    return sum_polynomials(
        (-1.0) ** k * turn_indicator(layout, k, direction) for k in range(i, j)
    )


def distance(layout: QubitLayout, i: int, j: int) -> BinaryPolynomial:
    """Return the turn-counting distance ``d(i, j) = sum_a Delta n_a(i, j)**2``.

    Quartic in the turn qubits: each indicator is quadratic, and squaring the signed
    counts multiplies two of them. This is the reason the dense encoding produces
    5-local terms once a contact qubit is applied, and the reason a genuine QUBO needs
    degree reduction.
    """
    if i > j:
        i, j = j, i
    total = BinaryPolynomial.zero()
    for direction in range(N_DIRECTIONS):
        component = delta_n(layout, i, j, direction)
        total = total + component * component
    return total


@dataclass(frozen=True)
class PenaltyWeights:
    """Penalty weights, every one of them derived rather than chosen.

    Attributes
    ----------
    back
        Weight of the back-tracking penalty in ``H_gc``.
    overlap
        Weight of the optional global self-avoidance penalty.
    lambda_1
        Per-pair weight forcing a claimed contact to really be at distance one.
    lambda_2
        Per-pair weight forbidding local overlaps near a claimed contact.
    energy_scale
        The bound the others are built from: the largest total interaction energy any
        conformation could collect.
    """

    back: float
    overlap: float
    lambda_1: Mapping[tuple[int, int], float]
    lambda_2: Mapping[tuple[int, int], float]
    energy_scale: float


def derive_penalties(
    instance: FoldingInstance, epsilon: ContactEnergy
) -> PenaltyWeights:
    """Derive every penalty weight from the instance's own energy scale.

    Derivation
    ----------
    Let ``E = sum |eps(i,j)|`` over all candidate contact pairs. No conformation can
    gain more than ``E`` from interactions, so any penalty strictly greater than ``E``
    cannot be paid for by collecting contacts. That gives

    ``lambda_back = lambda_2 = lambda_overlap = E + 1``.

    For the per-pair weight the paper states the condition directly (SI, after
    Eq. SI-30): the bracket of Eq. SI-30 must stay positive whenever beads ``i`` and
    ``j`` are not in contact, and since ``|N(i)| <= 3`` and
    ``-(j-i+1) < d(i,r) - d(i,j) < (j-i+1)``, this requires

    ``lambda_1 > 6 (j - i + 1) lambda_2 + eps(i,j)``.

    One is added to make the inequality strict in floating point. The weights therefore
    grow with chain separation, which is why they are stored per pair rather than
    globally: a single worst-case value would be correct but would needlessly roughen
    the landscape for close pairs, and a rougher landscape is harder for every solver.
    """
    pairs = contact_pairs(instance.n_beads)
    energy_scale = sum(abs(epsilon(i, j)) for i, j in pairs)

    base = energy_scale + 1.0
    lambda_2 = dict.fromkeys(pairs, base)
    lambda_1 = {
        (i, j): 6.0 * (j - i + 1) * lambda_2[(i, j)] + epsilon(i, j) + 1.0
        for i, j in pairs
    }

    for i, j in pairs:
        # Assert the paper's condition rather than trusting the arithmetic above.
        if lambda_1[(i, j)] <= 6.0 * (j - i + 1) * lambda_2[(i, j)] + epsilon(i, j):
            message = f"derived lambda_1 violates the bound for pair {(i, j)}"
            raise AssertionError(message)

    return PenaltyWeights(
        back=base,
        overlap=base,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        energy_scale=energy_scale,
    )


def build_growth_constraint(
    layout: QubitLayout, n_turns: int, weight: float
) -> BinaryPolynomial:
    """Return ``H_gc``, the back-tracking penalty of Eq. SI-16 and SI-17.

    ``T(i, j) = sum_a f_a(i) f_a(j)`` is 1 exactly when turns ``i`` and ``j`` choose the
    same direction index. Applied to consecutive turns this forbids back-tracking, which
    on the diamond lattice is a repeated index rather than an opposite one -- the
    property established in :mod:`foldq.lattice`.
    """
    terms = []
    for turn in range(n_turns - 1):
        overlap_indicator = sum_polynomials(
            turn_indicator(layout, turn, a) * turn_indicator(layout, turn + 1, a)
            for a in range(N_DIRECTIONS)
        )
        terms.append(weight * overlap_indicator)
    return sum_polynomials(terms)


def build_chirality_constraint(layout: QubitLayout) -> BinaryPolynomial:
    """Return ``H_ch``, which is identically zero for a main-chain-only model.

    Eq. SI-26 constrains the placement of the *first side-chain bead* relative to the
    two main-chain turns around its insertion point. This model gives every residue a
    single main-chain bead and no side chain, so there is no stereocentre to constrain
    and the term is empty.

    The function exists so that adding side chains is a change to this one place rather
    than a change to the shape of the Hamiltonian, and so that the absence of a
    chirality term is explicit rather than an omission a reader has to notice.
    """
    del layout
    return BinaryPolynomial.zero()


def build_contact_brackets(
    layout: QubitLayout,
    n_beads: int,
    epsilon: ContactEnergy,
    penalties: PenaltyWeights,
) -> dict[tuple[int, int], BinaryPolynomial]:
    """Return ``H_in`` for first-nearest-neighbour contacts, Eq. SI-29 and SI-30.

    For each candidate pair the contact qubit ``q_ij`` gates the term::

        q_ij ( eps_ij
               + lambda_1 (d(i,j) - 1)
               + sum_{r in N(j)} lambda_2 (2 - d(i,r))
               + sum_{m in N(i)} lambda_2 (2 - d(m,j)) )

    The first penalty makes claiming a contact unprofitable unless the beads really are
    at ``d == 1``. The other two prevent the local overlaps that could otherwise occur
    in the neighbourhood of a genuine contact: if ``i`` is in contact with ``j``, then
    ``i`` must not also be sitting on top of a bead bonded to ``j``.

    Note what this does *not* do. The overlap protection is local to a claimed contact,
    so a conformation can still fold through itself somewhere else entirely and pay
    nothing. That is a property of the published model, not an omission here; see
    :func:`build_self_avoidance` for the alternative and what it costs.
    """
    brackets: dict[tuple[int, int], BinaryPolynomial] = {}
    for i, j in layout.contact_qubits:
        lambda_1 = penalties.lambda_1[(i, j)]
        lambda_2 = penalties.lambda_2[(i, j)]

        bracket = BinaryPolynomial.constant(epsilon(i, j))
        bracket = bracket + lambda_1 * (distance(layout, i, j) - CONTACT_DISTANCE)
        for r in chain_neighbours(j, n_beads):
            bracket = bracket + lambda_2 * (2.0 - distance(layout, i, r))
        for m in chain_neighbours(i, n_beads):
            bracket = bracket + lambda_2 * (2.0 - distance(layout, m, j))

        brackets[(i, j)] = bracket
    return brackets


def build_interaction(
    layout: QubitLayout, brackets: Mapping[tuple[int, int], BinaryPolynomial]
) -> BinaryPolynomial:
    """Return ``H_in`` by gating each bracket with its contact qubit."""
    return sum_polynomials(
        BinaryPolynomial.variable(layout.contact_qubits[pair]) * bracket
        for pair, bracket in brackets.items()
    )


def overlap_patterns(n_steps: int) -> list[tuple[int, ...]]:
    """Return every turn pattern of ``n_steps`` whose net displacement is zero.

    A bead pair overlaps exactly when the sub-walk between them closes. Enumerating the
    closing patterns gives an exact indicator for that event, which is what
    :func:`build_self_avoidance` penalises.

    The diamond lattice has girth six, so there are no closing patterns at two or four
    steps other than immediate back-tracks, and those are already forbidden by
    ``H_gc``.
    """
    lattice = TetrahedralLattice()
    return [
        pattern
        for pattern in itertools.product(range(N_DIRECTIONS), repeat=n_steps)
        if lattice.walk(pattern)[-1] == (0, 0, 0)
    ]


def build_self_avoidance(
    layout: QubitLayout, n_beads: int, weight: float
) -> BinaryPolynomial:
    """Return ``H_saw``, an exact global excluded-volume penalty. Not in the paper.

    For every pair of beads on the same sublattice, an overlap means the sub-walk
    between them closes. The indicator for that is the sum, over every closing turn
    pattern, of the product of the corresponding turn indicators -- exact, and of degree
    ``2 * (j - i)`` in the qubits.

    Why the reference model omits this. The published Hamiltonian protects only against
    overlaps adjacent to a claimed contact, which keeps every term at most 5-local and
    the qubit count at ``O(N**2)``. Enforcing excluded volume globally costs high-degree
    terms that, after degree reduction, need far more auxiliary variables than the
    configuration register itself. That is affordable on classical or quantum-inspired
    hardware and not on a near-term quantum device -- which is a result worth measuring
    rather than a detail to bury, so both models are buildable here and the benchmark
    reports the cost.
    """
    terms = []
    for i in range(n_beads):
        for j in range(i + 2, n_beads, 2):
            for pattern in overlap_patterns(j - i):
                indicator = BinaryPolynomial.constant(1.0)
                for offset, direction in enumerate(pattern):
                    indicator = indicator * turn_indicator(
                        layout, i + offset, direction
                    )
                    if not indicator.terms:
                        break  # a fixed turn ruled this pattern out
                terms.append(weight * indicator)
    return sum_polynomials(terms)


@dataclass(frozen=True)
class QUBO:
    """A quadratic unconstrained binary optimisation problem."""

    matrix: NDArray[np.float64]
    offset: float

    def energy(self, bits: Sequence[int] | NDArray[np.int_]) -> float:
        """Return ``x^T Q x + offset`` for a binary vector."""
        x = np.asarray(bits, dtype=np.float64)
        return float(x @ self.matrix @ x + self.offset)

    @property
    def n_variables(self) -> int:
        """Number of binary variables."""
        return int(self.matrix.shape[0])


@dataclass(frozen=True)
class IsingModel:
    """An Ising model ``sum_{i<j} J_ij s_i s_j + sum_i h_i s_i + offset``."""

    couplings: NDArray[np.float64]
    fields: NDArray[np.float64]
    offset: float

    def energy(self, spins: Sequence[int] | NDArray[np.int_]) -> float:
        """Return the Ising energy of a spin vector with entries in ``{-1, +1}``."""
        s = np.asarray(spins, dtype=np.float64)
        return float(s @ self.couplings @ s + self.fields @ s + self.offset)

    @property
    def n_variables(self) -> int:
        """Number of spins."""
        return int(self.fields.shape[0])


@dataclass(frozen=True)
class FoldingHamiltonian:
    """The folding Hamiltonian and its equivalent representations.

    ``core`` is the single source of truth: an exact pseudo-Boolean polynomial over the
    turn and contact qubits, of whatever degree the physics requires. The quadratic
    views are derived from it, never written by hand.
    """

    instance: FoldingInstance
    layout: QubitLayout
    core: BinaryPolynomial
    constraints: BinaryPolynomial
    contact_brackets: Mapping[tuple[int, int], BinaryPolynomial]
    penalties: PenaltyWeights
    quadratized: QuadratizationResult
    enforce_global_saw: bool
    epsilon: ContactEnergy = field(repr=False)

    @property
    def core_degree(self) -> int:
        """Degree of the exact Hamiltonian before any reduction."""
        return self.core.degree

    def turn_assignment(self, turns: Sequence[int]) -> dict[int, int]:
        """Return the qubit assignment encoding a turn sequence.

        The first two turns are fixed by symmetry and carry no qubits, so they are
        required to match :data:`foldq.encoding.FIXED_TURNS` rather than being encoded.
        """
        if len(turns) != self.instance.n_turns:
            message = f"expected {self.instance.n_turns} turns, got {len(turns)}"
            raise ValueError(message)
        for turn in range(N_FIXED_TURNS):
            if turns[turn] != FIXED_TURNS[turn]:
                message = (
                    f"turn {turn} is fixed at {FIXED_TURNS[turn]} by symmetry, "
                    f"got {turns[turn]}"
                )
                raise ValueError(message)

        assignment: dict[int, int] = {}
        for turn in range(N_FIXED_TURNS, self.instance.n_turns):
            direction = turns[turn]
            assignment[self.layout.turn_qubits[(turn, 0)]] = direction >> 1
            assignment[self.layout.turn_qubits[(turn, 1)]] = direction & 1
        return assignment

    def energy_of_turns(
        self, turns: Sequence[int]
    ) -> tuple[float, dict[tuple[int, int], int]]:
        """Return the Hamiltonian's value for a conformation, and its contact qubits.

        Each contact qubit appears only as ``q_ij * bracket_ij``, and the bracket
        depends on the turn qubits alone. So for a fixed conformation the qubits are
        independent and the minimum is analytic: set ``q_ij = 1`` exactly when its
        bracket is negative. That makes the Hamiltonian's ground state reachable by
        sweeping conformations, without enumerating the contact and auxiliary registers.
        """
        assignment = self.turn_assignment(turns)
        total = self.constraints.evaluate(assignment)
        contacts: dict[tuple[int, int], int] = {}
        for pair, bracket in self.contact_brackets.items():
            value = bracket.evaluate(assignment)
            if value < 0.0:
                total += value
                contacts[pair] = 1
            else:
                contacts[pair] = 0
        return total, contacts

    def ground_state(self) -> tuple[float, tuple[int, ...], dict[tuple[int, int], int]]:
        """Return the Hamiltonian's exact ground state by sweeping conformations.

        Exhaustive over the symmetry-reduced turn space, and exact rather than
        variational, so it is the reference the QUBO and Ising minimisers are checked
        against.
        """
        best: tuple[float, tuple[int, ...], dict[tuple[int, int], int]] | None = None
        for turns in enumerate_turns(self.instance.n_turns, fix_symmetry=True):
            energy, contacts = self.energy_of_turns(turns)
            if best is None or energy < best[0]:
                best = (energy, turns, contacts)
        if best is None:  # pragma: no cover - enumerate_turns always yields
            message = "no conformations enumerated"
            raise RuntimeError(message)
        return best

    def to_qubo(self) -> QUBO:
        """Return the QUBO view, over primary qubits plus quadratization auxiliaries."""
        size = self.layout.total_qubits
        matrix = np.zeros((size, size), dtype=np.float64)
        offset = 0.0
        for monomial, coefficient in self.quadratized.polynomial.terms.items():
            indices = sorted(monomial)
            if not indices:
                offset += coefficient
            elif len(indices) == 1:
                matrix[indices[0], indices[0]] += coefficient
            else:
                matrix[indices[0], indices[1]] += coefficient
        return QUBO(matrix=matrix, offset=offset)

    def to_ising(self) -> IsingModel:
        """Return the Ising view, obtained from the QUBO by ``q = (1 - s) / 2``.

        Derived from the same quadratized polynomial as :meth:`to_qubo`, so the two are
        the same model in different coordinates rather than two constructions that have
        to be kept in step.
        """
        qubo = self.to_qubo()
        size = qubo.n_variables
        couplings = np.zeros((size, size), dtype=np.float64)
        fields = np.zeros(size, dtype=np.float64)
        offset = qubo.offset

        for i in range(size):
            diagonal = qubo.matrix[i, i]
            # q_i = (1 - s_i) / 2
            offset += diagonal / 2.0
            fields[i] -= diagonal / 2.0
            for j in range(i + 1, size):
                coefficient = qubo.matrix[i, j]
                if coefficient == 0.0:
                    continue
                # q_i q_j = (1 - s_i)(1 - s_j) / 4
                offset += coefficient / 4.0
                fields[i] -= coefficient / 4.0
                fields[j] -= coefficient / 4.0
                couplings[i, j] += coefficient / 4.0

        return IsingModel(couplings=couplings, fields=fields, offset=offset)

    def to_pauli(self) -> SparsePauliOp:
        """Return the exact Hamiltonian as a Qiskit ``SparsePauliOp``.

        Built from ``core`` directly, at its true degree, with no degree reduction
        and therefore no auxiliary qubits. Substituting ``q_i = (I - Z_i) / 2``
        turns each monomial into a sum of Pauli-Z strings, so the operator is
        diagonal -- as it must be for a classical cost function.
        """
        # Imported lazily: a purely classical benchmark run should not pay the
        # cost of importing qiskit.
        from qiskit.quantum_info import SparsePauliOp

        size = self.layout.n_primary_qubits
        accumulated: dict[frozenset[int], float] = {}
        for monomial, coefficient in self.core.terms.items():
            indices = sorted(monomial)
            # prod_i (I - Z_i)/2 expands over all subsets, with sign (-1)**|subset|.
            scale = coefficient / (2.0 ** len(indices))
            for size_of_subset in range(len(indices) + 1):
                for subset in itertools.combinations(indices, size_of_subset):
                    key = frozenset(subset)
                    accumulated[key] = accumulated.get(key, 0.0) + scale * (
                        (-1.0) ** size_of_subset
                    )

        sparse_list = [
            ("Z" * len(key) if key else "I", sorted(key) if key else [0], value)
            for key, value in accumulated.items()
            if abs(value) > 1e-12
        ]
        if not sparse_list:
            sparse_list = [("I", [0], 0.0)]
        return SparsePauliOp.from_sparse_list(sparse_list, num_qubits=size)


def build_hamiltonian(
    instance: FoldingInstance,
    *,
    epsilon: ContactEnergy | None = None,
    enforce_global_saw: bool = False,
    penalties: PenaltyWeights | None = None,
) -> FoldingHamiltonian:
    """Build the folding Hamiltonian for an instance.

    Parameters
    ----------
    instance
        The folding instance. Must be on the tetrahedral lattice, which is the only
        lattice the turn encoding is defined for.
    epsilon
        Contact energy model. Defaults to the HP model for the instance's sequence.
    enforce_global_saw
        When false (the default) the Hamiltonian reproduces the published model, whose
        overlap protection is local to claimed contacts. When true, an exact global
        excluded-volume term is added, so the ground state is guaranteed to decode to a
        self-avoiding conformation -- at a cost in degree and auxiliary variables that
        the benchmark reports.
    penalties
        Override the derived penalty weights. Intended for tests that check the derived
        bound is tight by deliberately violating it; production callers should leave
        this alone and let :func:`derive_penalties` do its job.
    """
    if instance.lattice.name != "tetrahedral":
        message = (
            "the turn encoding is defined for the tetrahedral lattice only, "
            f"got {instance.lattice.name!r}"
        )
        raise ValueError(message)
    if instance.n_turns < N_FIXED_TURNS:
        message = f"chain too short to encode: {instance.n_beads} beads"
        raise ValueError(message)

    epsilon = epsilon if epsilon is not None else hp_contact_energy(instance)
    layout = build_layout(instance)
    if penalties is None:
        penalties = derive_penalties(instance, epsilon)

    constraints = build_growth_constraint(
        layout, instance.n_turns, penalties.back
    ) + build_chirality_constraint(layout)
    if enforce_global_saw:
        constraints = constraints + build_self_avoidance(
            layout, instance.n_beads, penalties.overlap
        )

    brackets = build_contact_brackets(layout, instance.n_beads, epsilon, penalties)
    core = constraints + build_interaction(layout, brackets)

    quadratized = quadratize(core, first_auxiliary_index=layout.n_primary_qubits)

    return FoldingHamiltonian(
        instance=instance,
        layout=layout.with_auxiliaries(quadratized.n_auxiliaries),
        core=core,
        constraints=constraints,
        contact_brackets=brackets,
        penalties=penalties,
        quadratized=quadratized,
        enforce_global_saw=enforce_global_saw,
        epsilon=epsilon,
    )
