"""Tests for the benchmark metrics, especially their degenerate cases.

Every one of these quantities has an input for which the textbook formula produces
``inf`` or ``nan``, and a benchmark that prints either has stopped measuring anything.
"""

from __future__ import annotations

import math

import pytest

from foldq.benchmark.metrics import (
    approximation_ratio,
    success_probability,
    summarise,
    time_to_solution,
)


def test_success_probability_counts_runs_reaching_the_optimum() -> None:
    """The fraction of runs at or below the optimum, within tolerance."""
    assert success_probability([-2.0, -2.0, -1.0, 0.0], optimum=-2.0) == 0.5
    assert success_probability([-2.0, -2.0], optimum=-2.0) == 1.0
    assert success_probability([0.0, 1.0], optimum=-2.0) == 0.0


def test_success_probability_tolerates_floating_point_noise() -> None:
    """An energy a hair above the optimum still counts, since energies are computed."""
    assert success_probability([-2.0 + 1e-12], optimum=-2.0) == 1.0


def test_success_probability_rejects_an_empty_run_list() -> None:
    """Zero runs is a caller error, not a probability of zero."""
    with pytest.raises(ValueError, match="at least one run"):
        success_probability([], optimum=0.0)


def test_approximation_ratio_is_one_at_the_optimum() -> None:
    """Reaching the optimum gives a ratio of exactly 1."""
    assert approximation_ratio(-4.0, -4.0) == 1.0


def test_approximation_ratio_falls_below_one_as_quality_degrades() -> None:
    """With negative energies the ratio decreases as the answer gets worse.

    The opposite of the usual orientation, which is why RESULTS.md states the
    convention next to the table.
    """
    assert approximation_ratio(-2.0, -4.0) == 0.5
    assert approximation_ratio(0.0, -4.0) == 0.0


def test_approximation_ratio_handles_a_zero_optimum() -> None:
    """A zero optimum divides by zero, so the case is defined explicitly.

    A sequence with no hydrophobic contacts available has optimum zero. Finding zero is
    a perfect solve and scores 1.0; anything else makes the ratio meaningless, and None
    is returned rather than an invented number.
    """
    assert approximation_ratio(0.0, 0.0) == 1.0
    assert approximation_ratio(3.0, 0.0) is None


def test_time_to_solution_matches_the_formula() -> None:
    """TTS equals t_run ln(0.01) / ln(1 - p) for an interior probability."""
    expected = 2.0 * math.log(0.01) / math.log(0.5)
    assert time_to_solution(2.0, 0.5) == pytest.approx(expected)


def test_time_to_solution_with_certain_success_is_one_run() -> None:
    """p_s = 1 would divide by ln(0); one run already suffices."""
    assert time_to_solution(3.5, 1.0) == 3.5


def test_time_to_solution_is_undefined_when_the_optimum_is_never_reached() -> None:
    """p_s = 0 has no finite answer, so None is returned rather than inf.

    Reporting a very large number here would look like a measurement. It is not one:
    no number of repetitions reaches 99% confidence if the success rate is zero.
    """
    assert time_to_solution(3.5, 0.0) is None


def test_time_to_solution_improves_with_higher_success_probability() -> None:
    """A more reliable solver needs less time, holding run time fixed."""
    worse = time_to_solution(1.0, 0.2)
    better = time_to_solution(1.0, 0.8)
    assert worse is not None
    assert better is not None
    assert better < worse


@pytest.mark.parametrize(
    ("probability", "confidence", "wall_time"),
    [(-0.1, 0.99, 1.0), (1.1, 0.99, 1.0), (0.5, 0.0, 1.0), (0.5, 0.99, -1.0)],
)
def test_time_to_solution_rejects_out_of_range_inputs(
    probability: float, confidence: float, wall_time: float
) -> None:
    """Nonsense inputs raise rather than silently producing a number."""
    with pytest.raises(ValueError, match="must"):
        time_to_solution(wall_time, probability, confidence)


def test_summary_aggregates_repeated_runs() -> None:
    """A summary row carries every quantity the results table needs."""
    summary = summarise(
        solver="annealing",
        sequence="HHPHPPHH",
        encoding="dense",
        n_beads=8,
        optimum=-2.0,
        energies=[-2.0, -1.0, -2.0, 0.0],
        wall_times=[1.0, 1.0, 1.0, 1.0],
        evaluations=[10, 10, 10, 10],
    )
    assert summary.sequence == "HHPHPPHH"
    assert summary.success_probability == 0.5
    assert summary.best_energy == -2.0
    assert summary.approximation_ratio == 1.0
    assert summary.n_runs == 4
    assert summary.mean_evaluations == 10.0
    assert summary.time_to_solution_s is not None


def test_summary_reports_no_time_to_solution_when_never_solved() -> None:
    """A solver that never reaches the optimum has no TTS, and says so."""
    summary = summarise(
        solver="bifurcation",
        sequence="HHPHPPHH",
        encoding="one_hot",
        n_beads=8,
        optimum=-2.0,
        energies=[0.0, 0.0],
        wall_times=[1.0, 1.0],
        evaluations=[5, 5],
    )
    assert summary.success_probability == 0.0
    assert summary.time_to_solution_s is None


def test_summary_rejects_mismatched_input_lengths() -> None:
    """Per-run lists must line up, or the aggregation is meaningless."""
    with pytest.raises(ValueError, match="same length"):
        summarise(
            solver="annealing",
            sequence="HPHPPH",
            encoding="dense",
            n_beads=6,
            optimum=-1.0,
            energies=[-1.0],
            wall_times=[1.0, 1.0],
            evaluations=[1],
        )
