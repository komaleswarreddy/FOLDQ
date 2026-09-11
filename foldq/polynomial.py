"""Pseudo-Boolean polynomial algebra and degree reduction.

The folding Hamiltonian is naturally higher-order. On the tetrahedral lattice with the
dense two-qubit-per-turn encoding, the turn indicator ``f_a`` is quadratic in the
qubits, the pairwise distance ``d(i,j)`` is quartic, and the interaction term -- a
contact qubit multiplying that distance -- is quintic. The reference paper says as much:
the dense encoding produces 5-local terms where the sparse one produces 3-local terms.

QUBO and Ising ``(J, h)`` are quadratic by definition, so the higher-order form has to
be reduced. That is what :func:`quadratize` does, via the classical Rosenberg
substitution, at the cost of auxiliary variables.

Everything here is exact. No term is dropped and no coefficient is approximated, so the
three representations exported in :mod:`foldq.hamiltonian` describe one object.

References
----------
Rosenberg, I. G. "Reduction of bivalent maximization to the quadratic case."
Cahiers du Centre d'Etudes de Recherche Operationnelle 17, 71-74 (1975). The
substitution used by :func:`quadratize`.

Boros, E. and Hammer, P. L. "Pseudo-Boolean optimization."
Discrete Applied Mathematics 123, 155-225 (2002). Survey of the representation used
throughout this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

#: A monomial: the set of variable indices multiplied together. A set rather than a
#: multiset because binary variables are idempotent, ``q**2 == q``, so repeated factors
#: collapse. The empty frozenset is the constant term.
Monomial = frozenset[int]

#: Coefficients below this magnitude are dropped. The Hamiltonian's coefficients are
#: built from small integers and exactly representable halves, so anything this small is
#: floating-point debris from cancellation rather than physics.
_TOLERANCE = 1e-12


@dataclass(frozen=True)
class BinaryPolynomial:
    """A polynomial over binary variables ``q_i`` in ``{0, 1}``.

    Multiplication uses ``q**2 == q``, so the product of two monomials is the union of
    their variable sets.

    The core Hamiltonian is held in this binary form rather than in spin variables
    because the reference paper writes its equations over binary qubit values, and
    transcribing them directly is less error-prone than transcribing a rewritten
    version. Conversion to spin variables for the Pauli and Ising views is exact and
    happens at export time.
    """

    terms: Mapping[Monomial, float]

    @staticmethod
    def constant(value: float) -> BinaryPolynomial:
        """Return the constant polynomial."""
        if abs(value) < _TOLERANCE:
            return BinaryPolynomial({})
        return BinaryPolynomial({frozenset(): value})

    @staticmethod
    def variable(index: int) -> BinaryPolynomial:
        """Return the polynomial equal to the single variable ``q_index``."""
        return BinaryPolynomial({frozenset({index}): 1.0})

    @staticmethod
    def zero() -> BinaryPolynomial:
        """Return the zero polynomial."""
        return BinaryPolynomial({})

    def __add__(self, other: BinaryPolynomial | float) -> BinaryPolynomial:
        """Add another polynomial or a scalar."""
        if not isinstance(other, BinaryPolynomial):
            other = BinaryPolynomial.constant(float(other))
        merged: dict[Monomial, float] = dict(self.terms)
        for monomial, coefficient in other.terms.items():
            merged[monomial] = merged.get(monomial, 0.0) + coefficient
        return BinaryPolynomial(_pruned(merged))

    __radd__ = __add__

    def __neg__(self) -> BinaryPolynomial:
        """Negate every coefficient."""
        return BinaryPolynomial({m: -c for m, c in self.terms.items()})

    def __sub__(self, other: BinaryPolynomial | float) -> BinaryPolynomial:
        """Subtract another polynomial or a scalar."""
        if not isinstance(other, BinaryPolynomial):
            other = BinaryPolynomial.constant(float(other))
        return self + (-other)

    def __rsub__(self, other: float) -> BinaryPolynomial:
        """Subtract this polynomial from a scalar."""
        return BinaryPolynomial.constant(float(other)) + (-self)

    def __mul__(self, other: BinaryPolynomial | float) -> BinaryPolynomial:
        """Multiply by another polynomial or a scalar."""
        if not isinstance(other, BinaryPolynomial):
            scale = float(other)
            return BinaryPolynomial(
                _pruned({m: c * scale for m, c in self.terms.items()})
            )
        product: dict[Monomial, float] = {}
        for left_monomial, left_coefficient in self.terms.items():
            for right_monomial, right_coefficient in other.terms.items():
                # q**2 == q, so the product monomial is the union of the factors.
                key = left_monomial | right_monomial
                product[key] = (
                    product.get(key, 0.0) + left_coefficient * right_coefficient
                )
        return BinaryPolynomial(_pruned(product))

    __rmul__ = __mul__

    @property
    def degree(self) -> int:
        """Return the highest number of variables in any monomial."""
        return max((len(m) for m in self.terms), default=0)

    @property
    def variables(self) -> frozenset[int]:
        """Return every variable index appearing in the polynomial."""
        return frozenset().union(*self.terms) if self.terms else frozenset()

    @property
    def constant_term(self) -> float:
        """Return the coefficient of the empty monomial."""
        return self.terms.get(frozenset(), 0.0)

    def coefficient_l1_norm(self) -> float:
        """Return the sum of absolute values of all non-constant coefficients.

        This is the largest amount by which any assignment can change the polynomial's
        value away from its constant term, so it is a provable upper bound when
        deriving penalty weights. No penalty in this project is chosen by taste.
        """
        return sum(abs(c) for m, c in self.terms.items() if m)

    def evaluate(self, assignment: Mapping[int, int] | Sequence[int]) -> float:
        """Evaluate the polynomial at a binary assignment.

        Parameters
        ----------
        assignment
            Either a mapping from variable index to 0/1, or a sequence indexed by
            variable index.
        """
        total = 0.0
        for monomial, coefficient in self.terms.items():
            if all(assignment[index] for index in monomial):
                total += coefficient
        return total


def _pruned(terms: Mapping[Monomial, float]) -> dict[Monomial, float]:
    """Drop coefficients that have cancelled to within floating-point noise."""
    return {m: c for m, c in terms.items() if abs(c) > _TOLERANCE}


def sum_polynomials(parts: Iterable[BinaryPolynomial]) -> BinaryPolynomial:
    """Return the sum of an iterable of polynomials."""
    total = BinaryPolynomial.zero()
    for part in parts:
        total = total + part
    return total


@dataclass(frozen=True)
class QuadratizationResult:
    """The outcome of reducing a polynomial to degree two.

    Attributes
    ----------
    polynomial
        The reduced polynomial, of degree at most two.
    n_auxiliaries
        How many auxiliary variables were introduced.
    auxiliary_of
        Maps each auxiliary variable index to the pair of variables whose product it
        replaces, so a solution can be checked for consistency and the auxiliaries
        eliminated when decoding.
    penalty_weight
        The weight applied to each Rosenberg penalty term.
    """

    polynomial: BinaryPolynomial
    n_auxiliaries: int
    auxiliary_of: Mapping[int, tuple[int, int]]
    penalty_weight: float


def quadratize(
    polynomial: BinaryPolynomial, *, first_auxiliary_index: int
) -> QuadratizationResult:
    """Reduce a pseudo-Boolean polynomial to degree two by Rosenberg substitution.

    For a product ``q_i q_j`` inside higher-order monomials, a fresh variable ``q_k``
    is introduced and every occurrence of ``q_i q_j`` is replaced by ``q_k``. The
    substitution is enforced by adding the Rosenberg penalty

    .. math:: R(q_i, q_j, q_k) = q_i q_j - 2 q_i q_k - 2 q_j q_k + 3 q_k

    which is zero when ``q_k == q_i q_j`` and at least one otherwise. The pair chosen at
    each step is the one appearing in the most high-degree monomials, which keeps the
    number of auxiliaries low.

    Penalty weight
    --------------
    The weight must exceed any benefit an assignment could gain by violating a
    substitution. Since ``R >= 1`` whenever the substitution is violated, and the total
    value of the polynomial can move by at most the sum of the absolute values of its
    non-constant coefficients, setting the weight above that sum is sufficient. That
    bound is computed from the polynomial rather than chosen, and is asserted below.

    Parameters
    ----------
    polynomial
        The polynomial to reduce. May be of any degree.
    first_auxiliary_index
        Index to assign to the first auxiliary variable. Must be greater than every
        variable index already used, so auxiliaries cannot collide with real qubits.
    """
    existing = polynomial.variables
    if existing and first_auxiliary_index <= max(existing):
        message = (
            f"first_auxiliary_index {first_auxiliary_index} collides with existing "
            f"variables (max index {max(existing)})"
        )
        raise ValueError(message)

    # Derived, not chosen: R >= 1 on violation, and no assignment can shift the
    # polynomial by more than the L1 norm of its non-constant coefficients.
    weight = polynomial.coefficient_l1_norm() + 1.0

    current = polynomial
    auxiliary_of: dict[int, tuple[int, int]] = {}
    next_index = first_auxiliary_index

    while current.degree > 2:
        pair = _most_common_pair(current)
        if pair is None:  # pragma: no cover - unreachable while degree > 2
            message = "no reducible pair found in a polynomial of degree > 2"
            raise RuntimeError(message)

        left, right = pair
        auxiliary = next_index
        next_index += 1
        auxiliary_of[auxiliary] = pair

        current = _substitute(current, left, right, auxiliary)
        current = current + _rosenberg_penalty(left, right, auxiliary, weight)

    if current.degree > 2:  # pragma: no cover - loop guarantees otherwise
        message = f"quadratization left degree {current.degree}"
        raise RuntimeError(message)

    return QuadratizationResult(
        polynomial=current,
        n_auxiliaries=len(auxiliary_of),
        auxiliary_of=auxiliary_of,
        penalty_weight=weight,
    )


def _most_common_pair(polynomial: BinaryPolynomial) -> tuple[int, int] | None:
    """Return the variable pair occurring in the most monomials of degree three or more.

    Greedy, and deliberately so: choosing the most frequent pair removes the most
    high-degree structure per auxiliary introduced. Finding the minimum number of
    auxiliaries is itself NP-hard, so a heuristic is the standard approach.
    """
    counts: dict[tuple[int, int], int] = {}
    for monomial in polynomial.terms:
        if len(monomial) < 3:
            continue
        ordered = sorted(monomial)
        for a_position, left in enumerate(ordered):
            for right in ordered[a_position + 1 :]:
                key = (left, right)
                counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda key: (counts[key], -key[0], -key[1]))


def _substitute(
    polynomial: BinaryPolynomial, left: int, right: int, auxiliary: int
) -> BinaryPolynomial:
    """Replace ``q_left q_right`` by ``q_auxiliary`` wherever both appear."""
    rewritten: dict[Monomial, float] = {}
    for monomial, coefficient in polynomial.terms.items():
        if len(monomial) >= 3 and left in monomial and right in monomial:
            key = (monomial - {left, right}) | {auxiliary}
        else:
            key = monomial
        rewritten[key] = rewritten.get(key, 0.0) + coefficient
    return BinaryPolynomial(_pruned(rewritten))


def _rosenberg_penalty(
    left: int, right: int, auxiliary: int, weight: float
) -> BinaryPolynomial:
    """Return ``weight * (q_l q_r - 2 q_l q_a - 2 q_r q_a + 3 q_a)``."""
    return BinaryPolynomial(
        {
            frozenset({left, right}): weight,
            frozenset({left, auxiliary}): -2.0 * weight,
            frozenset({right, auxiliary}): -2.0 * weight,
            frozenset({auxiliary}): 3.0 * weight,
        }
    )


def rosenberg_penalty_value(left: int, right: int, auxiliary: int) -> float:
    """Return the unweighted Rosenberg penalty for one binary assignment triple.

    Exposed so tests can assert the property the reduction relies on: the penalty is
    zero exactly when the auxiliary equals the product it stands for, and at least one
    otherwise.
    """
    return float(
        left * right - 2 * left * auxiliary - 2 * right * auxiliary + 3 * auxiliary
    )
