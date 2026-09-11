"""Metrics for comparing stochastic solvers.

Three quantities do the work: how often a solver reaches the exact optimum, how good
its answer is when it does not, and how long it would take to be confident of success.
Each has an edge case that produces ``nan`` or ``inf`` if written naively, and each is
handled explicitly here rather than left to float arithmetic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

#: Confidence level for time-to-solution. The convention in the Ising-machine
#: literature: the time needed to have seen the optimum at least once with this
#: probability.
TTS_CONFIDENCE = 0.99

#: Energies within this of each other count as equal. Energies here are sums of small
#: integers, so this only absorbs floating-point noise from the QUBO evaluation.
ENERGY_TOLERANCE = 1e-9


def success_probability(
    energies: Sequence[float], optimum: float, tolerance: float = ENERGY_TOLERANCE
) -> float:
    """Return the fraction of runs that reached the exact optimum.

    Parameters
    ----------
    energies
        Best energy from each independent run.
    optimum
        The exact optimum, from exhaustive enumeration.
    tolerance
        Energies within this of the optimum count as having reached it.
    """
    if not energies:
        message = "success probability needs at least one run"
        raise ValueError(message)
    hits = sum(1 for energy in energies if energy <= optimum + tolerance)
    return hits / len(energies)


def approximation_ratio(
    found: float, optimum: float, tolerance: float = ENERGY_TOLERANCE
) -> float | None:
    """Return ``E_found / E_optimum``, or ``None`` where the ratio is undefined.

    The brief's formula, kept so the numbers stay comparable with the literature, but
    it needs care here. Energies in the HP model are negative or zero, so:

    * When the optimum is zero -- a sequence with no hydrophobic contacts available --
      the ratio divides by zero. If the solver also found zero it has solved the
      instance perfectly, and the ratio is defined to be 1.0. If it found anything else
      the quantity is meaningless, and ``None`` is returned rather than an invented
      number.
    * For a negative optimum the ratio runs *below* 1 as the answer gets worse, which
      inverts the usual reading. 1.0 is optimal and smaller is worse.

    RESULTS.md states this convention alongside the table, because a reader who assumes
    the usual "higher is worse" orientation will read every row backwards.
    """
    if abs(optimum) <= tolerance:
        if abs(found) <= tolerance:
            return 1.0
        return None
    return found / optimum


def time_to_solution(
    wall_time_per_run: float,
    probability: float,
    confidence: float = TTS_CONFIDENCE,
) -> float | None:
    """Return the time to reach the optimum at least once with ``confidence``.

    ``TTS = t_run * ln(1 - confidence) / ln(1 - p_s)``

    This is the right metric for comparing stochastic optimisers, because wall-clock
    time alone rewards a solver that returns fast answers that are usually wrong. TTS
    folds quality and speed into one number.

    Edge cases, handled rather than left to produce ``inf`` or ``nan``:

    * ``p_s == 1``: every run succeeds, so one run suffices and TTS is ``t_run``. The
      formula would divide by ``ln(0)``.
    * ``p_s == 0``: the optimum was never reached, so no finite number of repetitions
      gives the required confidence. ``None`` is returned, and the caller is expected to
      report "not reached" rather than a large number that looks like a measurement.
    """
    if not 0.0 <= probability <= 1.0:
        message = f"probability must lie in [0, 1], got {probability}"
        raise ValueError(message)
    if not 0.0 < confidence < 1.0:
        message = f"confidence must lie in (0, 1), got {confidence}"
        raise ValueError(message)
    if wall_time_per_run < 0.0:
        message = f"wall_time_per_run must be non-negative, got {wall_time_per_run}"
        raise ValueError(message)

    if probability >= 1.0:
        return wall_time_per_run
    if probability <= 0.0:
        return None
    return wall_time_per_run * math.log(1.0 - confidence) / math.log(1.0 - probability)


@dataclass(frozen=True)
class SolverSummary:
    """Aggregated results for one solver on one instance.

    Attributes
    ----------
    solver
        Solver name.
    sequence
        The HP sequence folded. Two different sequences can share a chain length, so
        without this a results table cannot tell their rows apart.
    encoding
        Turn encoding the Hamiltonian used.
    n_beads
        Chain length.
    optimum
        Exact optimum from exhaustive enumeration.
    best_energy
        Best energy found across all runs.
    success_probability
        Fraction of runs reaching the optimum.
    approximation_ratio
        Ratio for the best energy found, or ``None`` where undefined.
    mean_wall_time_s
        Mean wall-clock time of a single run.
    time_to_solution_s
        Time to reach the optimum with 99% confidence, or ``None`` if never reached.
    n_runs
        Number of independent seeded runs.
    mean_evaluations
        Mean objective evaluations per run.
    """

    solver: str
    sequence: str
    encoding: str
    n_beads: int
    optimum: float
    best_energy: float
    success_probability: float
    approximation_ratio: float | None
    mean_wall_time_s: float
    time_to_solution_s: float | None
    n_runs: int
    mean_evaluations: float


def summarise(
    solver: str,
    sequence: str,
    encoding: str,
    n_beads: int,
    optimum: float,
    energies: Sequence[float],
    wall_times: Sequence[float],
    evaluations: Sequence[int],
) -> SolverSummary:
    """Aggregate the results of repeated seeded runs into one summary row."""
    if not energies:
        message = "cannot summarise zero runs"
        raise ValueError(message)
    if not len(energies) == len(wall_times) == len(evaluations):
        message = "energies, wall_times and evaluations must be the same length"
        raise ValueError(message)

    probability = success_probability(energies, optimum)
    mean_time = sum(wall_times) / len(wall_times)
    best = min(energies)

    return SolverSummary(
        solver=solver,
        sequence=sequence,
        encoding=encoding,
        n_beads=n_beads,
        optimum=optimum,
        best_energy=best,
        success_probability=probability,
        approximation_ratio=approximation_ratio(best, optimum),
        mean_wall_time_s=mean_time,
        time_to_solution_s=time_to_solution(mean_time, probability),
        n_runs=len(energies),
        mean_evaluations=sum(evaluations) / len(evaluations),
    )
