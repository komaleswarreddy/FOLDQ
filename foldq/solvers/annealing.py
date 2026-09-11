"""Simulated annealing on the QUBO form of the folding Hamiltonian.

The classical baseline. It solves the *same* Hamiltonian instance as every other
solver -- the QUBO exported by :meth:`foldq.hamiltonian.FoldingHamiltonian.to_qubo`,
auxiliary variables included -- because a benchmark across solvers that were given
different problems measures nothing.

Temperatures are derived from the instance rather than chosen. See
:func:`derive_schedule` for the derivation.

References
----------
Kirkpatrick, S., Gelatt, C. D. and Vecchi, M. P. "Optimization by simulated
annealing." Science 220, 671-680 (1983).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from foldq.hamiltonian import QUBO, FoldingHamiltonian
from foldq.solvers.base import SolverResult

#: Acceptance probability targeted for the worst uphill move at the start of the run.
#: One half means the chain begins almost freely diffusing, which is the standard
#: condition for the initial temperature of a geometric schedule.
_START_ACCEPTANCE = 0.5

#: Acceptance probability targeted for the *smallest* uphill move at the end of the
#: run. One in a hundred means the chain is effectively frozen but not completely
#: stuck, so the final sweeps still perform greedy repair.
_END_ACCEPTANCE = 0.01


@dataclass(frozen=True)
class AnnealingConfig:
    """Configuration for a simulated annealing run.

    Attributes
    ----------
    n_sweeps
        Number of sweeps. One sweep proposes as many moves as there are variables, so
        the total work is ``n_sweeps * n_variables``.
    two_bit_move_probability
        Fraction of proposals that flip two variables at once. Single-bit moves alone
        cannot cross the barriers created by the penalty terms, because flipping a turn
        qubit without its partner usually lands on an invalid turn; two-bit moves let
        the chain change a whole turn in one step.
    slave_auxiliaries
        Propose moves only on the primary variables and recompute the quadratization
        auxiliaries from them after every proposal, instead of treating all variables
        as independent.

        This is not a different problem -- the objective is the same QUBO -- but it is a
        far better move set for it. Auxiliaries are *determined* by the primaries, so
        searching them independently wastes the search on configurations that the
        Rosenberg penalties will reject anyway, and those penalties are orders of
        magnitude larger than the contact energies the search is actually trying to
        resolve. Slaving them shrinks the effective search space from every variable to
        the primaries alone.
    """

    n_sweeps: int = 2000
    two_bit_move_probability: float = 0.5
    slave_auxiliaries: bool = False


@dataclass(frozen=True)
class AnnealingSchedule:
    """A geometric cooling schedule, derived from the problem's own energy scales."""

    beta_start: float
    beta_end: float
    n_sweeps: int

    def betas(self) -> NDArray[np.float64]:
        """Return the inverse temperature for each sweep.

        Geometric in temperature, so uniform in ``log T`` -- the schedule that spends
        equal effort on each decade of energy scale, which is what makes it the
        standard choice rather than a linear ramp.
        """
        if self.n_sweeps == 1:
            return np.array([self.beta_end])
        ratio = self.beta_end / self.beta_start
        exponent = np.linspace(0.0, 1.0, self.n_sweeps)
        return self.beta_start * ratio**exponent


def derive_schedule(qubo: QUBO, n_sweeps: int) -> AnnealingSchedule:
    """Derive start and end temperatures from the QUBO's coefficients.

    Derivation
    ----------
    Flipping variable ``i`` changes the energy by at most

    ``dE_max(i) = |Q_ii| + sum_j |W_ij|``

    where ``W`` is the symmetrised coupling matrix, since every neighbouring variable
    can contribute at most its coupling. The largest such value over ``i`` is the
    steepest uphill move the chain can ever face, and the initial temperature is set so
    that it is accepted with probability :data:`_START_ACCEPTANCE`::

        T_start = dE_max / ln(1 / _START_ACCEPTANCE)

    At the other end, the finest distinction the chain has to resolve is the smallest
    non-zero coefficient magnitude, and the final temperature makes *that* move
    accepted with probability :data:`_END_ACCEPTANCE`::

        T_end = dE_min / ln(1 / _END_ACCEPTANCE)

    Both come from the matrix that was handed in, so a Hamiltonian with larger penalty
    weights automatically gets a hotter start. Nothing here is tuned.
    """
    symmetric = _symmetrise(qubo.matrix)
    diagonal = np.abs(np.diag(qubo.matrix))
    off_diagonal = np.abs(symmetric).sum(axis=1) - np.abs(np.diag(symmetric))
    delta_max = float(np.max(diagonal + off_diagonal))

    magnitudes = np.abs(qubo.matrix[qubo.matrix != 0.0])
    delta_min = float(np.min(magnitudes)) if magnitudes.size else 1.0

    if delta_max <= 0.0:
        delta_max = 1.0
    delta_min = min(delta_min, delta_max)

    temperature_start = delta_max / np.log(1.0 / _START_ACCEPTANCE)
    temperature_end = delta_min / np.log(1.0 / _END_ACCEPTANCE)
    temperature_end = min(temperature_end, temperature_start)

    return AnnealingSchedule(
        beta_start=1.0 / temperature_start,
        beta_end=1.0 / temperature_end,
        n_sweeps=n_sweeps,
    )


def _symmetrise(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the symmetric coupling matrix with the diagonal preserved.

    The QUBO is stored upper triangular. Symmetrising halves each off-diagonal entry
    into both positions so that a single matrix-vector product gives every variable's
    local field.
    """
    upper = np.triu(matrix, k=1)
    symmetric = upper + upper.T
    np.fill_diagonal(symmetric, np.diag(matrix))
    return symmetric


def anneal(
    qubo: QUBO,
    seed: int,
    config: AnnealingConfig | None = None,
) -> tuple[float, NDArray[np.int_], int]:
    """Run simulated annealing on a QUBO and return the best state found.

    Returns the best energy, the bit vector achieving it, and the number of proposals
    evaluated. Fully determined by ``seed``: two runs with the same seed produce
    identical output.
    """
    config = config or AnnealingConfig()
    rng = np.random.default_rng(seed)

    size = qubo.n_variables
    symmetric = _symmetrise(qubo.matrix)
    diagonal = np.diag(qubo.matrix).copy()
    couplings = symmetric.copy()
    np.fill_diagonal(couplings, 0.0)

    state = rng.integers(0, 2, size=size).astype(np.int64)
    # Local field: the energy change of a flip is (1 - 2x_i) * field_i.
    field = diagonal + couplings @ state
    energy = qubo.energy(state)

    best_energy = energy
    best_state = state.copy()
    evaluations = 0

    for beta in derive_schedule(qubo, config.n_sweeps).betas():
        for _ in range(size):
            evaluations += 1
            use_pair = size > 1 and rng.random() < config.two_bit_move_probability
            if use_pair:
                i, j = rng.choice(size, size=2, replace=False)
                delta = (
                    (1 - 2 * state[i]) * field[i]
                    + (1 - 2 * state[j]) * field[j]
                    + (1 - 2 * state[i]) * (1 - 2 * state[j]) * couplings[i, j]
                )
                flips: tuple[int, ...] = (int(i), int(j))
            else:
                i = int(rng.integers(size))
                delta = (1 - 2 * state[i]) * field[i]
                flips = (i,)

            if delta <= 0.0 or rng.random() < np.exp(-beta * delta):
                for index in flips:
                    state[index] = 1 - state[index]
                    # Sign of the change in x_index, applied to every neighbour.
                    direction = 1 if state[index] == 1 else -1
                    field += direction * couplings[:, index]
                energy += delta
                if energy < best_energy:
                    best_energy = energy
                    best_state = state.copy()

    return best_energy, best_state, evaluations


def anneal_slaved(
    hamiltonian: FoldingHamiltonian,
    seed: int,
    config: AnnealingConfig | None = None,
) -> tuple[float, NDArray[np.int_], int]:
    """Anneal over the primary variables, recomputing auxiliaries after each proposal.

    Same objective as :func:`anneal` -- the QUBO exported by the Hamiltonian -- but the
    chain only ever proposes flips of turn and contact qubits. Auxiliaries are repaired
    to the products they stand for before the energy is measured, which is provably
    non-worsening and removes the Rosenberg penalties from the landscape the chain has
    to cross.
    """
    config = config or AnnealingConfig()
    rng = np.random.default_rng(seed)
    qubo = hamiltonian.to_qubo()

    n_primary = hamiltonian.layout.n_primary_qubits
    state = np.zeros(qubo.n_variables, dtype=np.int64)
    state[:n_primary] = rng.integers(0, 2, size=n_primary)
    state = hamiltonian.repair_auxiliaries(state)
    energy = qubo.energy(state)

    best_energy = energy
    best_state = state.copy()
    evaluations = 0

    for beta in derive_schedule(qubo, config.n_sweeps).betas():
        for _ in range(n_primary):
            evaluations += 1
            if n_primary > 1 and rng.random() < config.two_bit_move_probability:
                flips = rng.choice(n_primary, size=2, replace=False)
            else:
                flips = rng.integers(n_primary, size=1)

            candidate = state.copy()
            candidate[flips] ^= 1
            candidate = hamiltonian.repair_auxiliaries(candidate)
            candidate_energy = qubo.energy(candidate)

            delta = candidate_energy - energy
            if delta <= 0.0 or rng.random() < np.exp(-beta * delta):
                state, energy = candidate, candidate_energy
                if energy < best_energy:
                    best_energy = energy
                    best_state = state.copy()

    return best_energy, best_state, evaluations


class AnnealingSolver:
    """Simulated annealing, wrapped in the :class:`~foldq.solvers.base.Solver` API."""

    def __init__(self, config: AnnealingConfig | None = None) -> None:
        """Store the run configuration."""
        self._config = config or AnnealingConfig()

    @property
    def name(self) -> str:
        """Short identifier used in artifacts, tables and figures."""
        return "annealing"

    def solve(
        self, hamiltonian: FoldingHamiltonian, seed: int | None = None
    ) -> SolverResult:
        """Anneal the Hamiltonian's QUBO form and decode the result."""
        if seed is None:
            message = "simulated annealing requires an explicit seed"
            raise ValueError(message)

        qubo = hamiltonian.to_qubo()
        start = time.perf_counter()
        if self._config.slave_auxiliaries:
            _, state, evaluations = anneal_slaved(hamiltonian, seed, self._config)
        else:
            _, state, evaluations = anneal(qubo, seed, self._config)
        # Repairing auxiliaries is provably non-worsening; see
        # FoldingHamiltonian.repair_auxiliaries.
        state = hamiltonian.repair_auxiliaries(state)
        energy = qubo.energy(state)
        elapsed = time.perf_counter() - start

        turns = hamiltonian.decode_turns(state)
        return SolverResult(
            energy=energy,
            turns=turns,
            n_evaluations=evaluations,
            wall_time_s=elapsed,
            seed=seed,
            metadata={
                "solver": self.name,
                "n_variables": qubo.n_variables,
                "n_sweeps": self._config.n_sweeps,
                "two_bit_move_probability": self._config.two_bit_move_probability,
                "slave_auxiliaries": self._config.slave_auxiliaries,
            },
        )
