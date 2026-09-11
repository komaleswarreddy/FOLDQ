"""Tests for the pseudo-Boolean algebra and Rosenberg degree reduction.

The guarantee that matters is not that quadratization produces *a* quadratic
polynomial, but that minimising the reduced polynomial over its auxiliary variables
reproduces the original value at every assignment of the real variables. Everything
downstream -- the QUBO handed to simulated annealing, the Ising model handed to
simulated bifurcation -- rests on that.
"""

from __future__ import annotations

import itertools

import pytest

from foldq.polynomial import (
    BinaryPolynomial,
    quadratize,
    rosenberg_penalty_value,
    sum_polynomials,
)


def test_binary_variables_are_idempotent() -> None:
    """``q * q == q``, so a monomial is a set of indices rather than a multiset."""
    q0 = BinaryPolynomial.variable(0)
    assert (q0 * q0).terms == q0.terms


def test_algebra_matches_direct_evaluation() -> None:
    """Sums and products evaluate the same as the arithmetic they stand for."""
    q0 = BinaryPolynomial.variable(0)
    q1 = BinaryPolynomial.variable(1)
    expression = (1 - q0) * (1 - q1) + 3.0 * q0 * q1 - 2.0
    for bits in itertools.product((0, 1), repeat=2):
        expected = (1 - bits[0]) * (1 - bits[1]) + 3.0 * bits[0] * bits[1] - 2.0
        assert expression.evaluate(bits) == pytest.approx(expected)


def test_degree_and_variables_are_reported() -> None:
    """Degree is the largest monomial size and variables are the union of indices."""
    poly = BinaryPolynomial.variable(0) * BinaryPolynomial.variable(
        1
    ) * BinaryPolynomial.variable(2) + BinaryPolynomial.variable(5)
    assert poly.degree == 3
    assert poly.variables == frozenset({0, 1, 2, 5})


def test_cancellation_removes_terms_entirely() -> None:
    """Terms that cancel are dropped, so degree does not report phantom structure."""
    q0 = BinaryPolynomial.variable(0)
    assert (q0 - q0).terms == {}
    assert (q0 - q0).degree == 0


def test_l1_norm_bounds_the_achievable_swing() -> None:
    """No assignment moves the polynomial further from its constant than the L1 norm.

    This is the bound every derived penalty weight in the project rests on, so it is
    checked exhaustively rather than argued.
    """
    q = [BinaryPolynomial.variable(i) for i in range(3)]
    poly = 2.0 * q[0] - 3.0 * q[1] * q[2] + 1.5 * q[0] * q[1] * q[2] + 7.0
    bound = poly.coefficient_l1_norm()
    for bits in itertools.product((0, 1), repeat=3):
        assert abs(poly.evaluate(bits) - poly.constant_term) <= bound + 1e-12


@pytest.mark.parametrize(
    ("left", "right", "auxiliary"), list(itertools.product((0, 1), repeat=3))
)
def test_rosenberg_penalty_is_zero_exactly_when_consistent(
    left: int, right: int, auxiliary: int
) -> None:
    """R is 0 when the auxiliary equals the product, and at least 1 otherwise.

    The whole reduction depends on this: it is what lets a finite penalty weight force
    the substitution to hold in every minimiser.
    """
    value = rosenberg_penalty_value(left, right, auxiliary)
    if auxiliary == left * right:
        assert value == 0
    else:
        assert value >= 1


def _minimum_over_auxiliaries(
    reduced: BinaryPolynomial, assignment: tuple[int, ...], auxiliary_indices: list[int]
) -> float:
    """Return the reduced polynomial's minimum over all settings of its auxiliaries."""
    best = float("inf")
    for auxiliary_bits in itertools.product((0, 1), repeat=len(auxiliary_indices)):
        full = dict(enumerate(assignment))
        full.update(dict(zip(auxiliary_indices, auxiliary_bits, strict=True)))
        best = min(best, reduced.evaluate(full))
    return best


@pytest.mark.parametrize("n_vars", [3, 4, 5])
def test_quadratization_preserves_value_after_minimising_auxiliaries(
    n_vars: int,
) -> None:
    """Minimising over auxiliaries reproduces the original at every assignment.

    This is the exact statement of what Rosenberg reduction guarantees, and it is
    checked over the whole assignment space rather than at the optimum alone -- a
    reduction that agreed only at the minimum would still corrupt every solver that
    walks the landscape.
    """
    q = [BinaryPolynomial.variable(i) for i in range(n_vars)]
    original = sum_polynomials(
        [
            2.0 * q[0] * q[1] * q[2],
            -3.0 * q[-1] * q[-2],
            1.0 * sum_polynomials(q),
            BinaryPolynomial.constant(-1.0),
        ]
    )
    if n_vars >= 4:
        original = original - 4.0 * q[0] * q[1] * q[2] * q[3]

    result = quadratize(original, first_auxiliary_index=n_vars)
    assert result.polynomial.degree <= 2
    assert result.n_auxiliaries >= 1

    auxiliary_indices = sorted(result.auxiliary_of)
    for bits in itertools.product((0, 1), repeat=n_vars):
        assert _minimum_over_auxiliaries(
            result.polynomial, bits, auxiliary_indices
        ) == pytest.approx(original.evaluate(bits))


def test_quadratization_of_an_already_quadratic_polynomial_is_a_no_op() -> None:
    """Nothing is introduced when the polynomial is already of degree two."""
    q0, q1 = BinaryPolynomial.variable(0), BinaryPolynomial.variable(1)
    result = quadratize(3.0 * q0 * q1 - q0, first_auxiliary_index=2)
    assert result.n_auxiliaries == 0
    assert result.polynomial.terms == (3.0 * q0 * q1 - q0).terms


def test_auxiliary_indices_may_not_collide_with_real_variables() -> None:
    """Starting auxiliaries below an existing index is rejected, not silently merged."""
    q = [BinaryPolynomial.variable(i) for i in range(3)]
    cubic = q[0] * q[1] * q[2]
    with pytest.raises(ValueError, match="collides"):
        quadratize(cubic, first_auxiliary_index=1)
