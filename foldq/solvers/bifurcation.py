"""Ballistic simulated bifurcation: the quantum-inspired solver.

Simulated bifurcation is classical mechanics, not quantum mechanics. It descends from
the quantum bifurcation machine -- a network of Kerr-nonlinear parametric oscillators
driven through a bifurcation, which solves the Ising problem by quantum adiabatic
evolution. Taking the classical limit of that system's Hamiltonian yields equations of
motion that can be integrated on ordinary hardware, and which keep the property that
made the quantum system attractive: every oscillator's update depends only on a
matrix-vector product, so all N variables can be updated *simultaneously*.

That is the whole point, and the reason the algorithm is relevant to quantum-inspired
silicon. Simulated annealing must update spins sequentially, because accepting a flip
changes the local fields that the next proposal depends on. Simulated bifurcation has
no such dependency, so it maps onto wide parallel hardware -- GPUs, FPGAs, or custom
accelerators -- in a way annealing does not.

Equations of motion
-------------------
Ballistic SB, from Eqs. (2)-(4) of Kanao and Goto (2022), restating Goto et al. (2021)::

    dx_i/dt = a_0 y_i
    dy_i/dt = -[a_0 - a(t)] x_i + c_0 f_i
    f_i     = sum_j J_ij x_j

integrated with the symplectic Euler method, with the control parameter ``a(t)``
increased linearly from zero to ``a_0``. Positions are confined to ``|x_i| <= 1`` by
perfectly inelastic walls: after each step, any ``|x_i| > 1`` is set to ``sgn(x_i)``
with ``y_i`` set to zero. The walls are what distinguishes *ballistic* SB from the
original adiabatic form, which instead used a quartic restoring term; they remove the
slow high-energy excursions and give much faster convergence.

The spins are read off at the end as ``s_i = sgn(x_i)``.

References
----------
Goto, H., Endo, K., Suzuki, M., Sakai, Y., Kanao, T., Hamakawa, Y., Hidaka, R.,
Yamasaki, M. and Tatsumura, K. "High-performance combinatorial optimization based on
classical mechanics." Science Advances 7, eabe7953 (2021). Introduces ballistic and
discrete SB.

Kanao, T. and Goto, H. "Simulated bifurcation assisted by thermal fluctuation."
arXiv:2203.08361 (2022). Eqs. (2)-(5) and (12), the form transcribed here.

Goto, H., Tatsumura, K. and Dixon, A. R. "Combinatorial optimization by simulating
adiabatic bifurcations in nonlinear Hamiltonian systems." Science Advances 5,
eaav2372 (2019). The original simulated bifurcation algorithm.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from foldq.hamiltonian import FoldingHamiltonian, IsingModel
from foldq.solvers.base import SolverResult


@dataclass(frozen=True)
class BifurcationConfig:
    """Configuration for a ballistic simulated bifurcation run.

    The defaults are the values reported for bSB in Kanao and Goto (2022), Fig. 1, and
    are exposed rather than hidden because they are empirical constants from the
    literature, not quantities derivable from this problem. ``c_0`` itself *is* derived
    from the instance -- see :func:`derive_coupling_strength`.

    Attributes
    ----------
    n_steps
        Number of symplectic Euler steps. Also the length of the ``a(t)`` ramp.
    dt
        Integration time step. 0.7 is the value reported for bSB.
    c1
        Dimensionless coupling constant, tuned around 0.5 on the basis of random matrix
        theory; 0.6 is the reported bSB value.
    a0
        Final value of the control parameter, and the detuning. Fixed at 1 throughout
        the literature, which sets the unit of time.
    n_replicas
        Independent trajectories integrated simultaneously. This is the axis that makes
        the algorithm parallel; every replica shares the coupling matrix, so a batch
        costs one matrix-matrix product per step instead of ``n_replicas`` separate
        matrix-vector products.
    evaluate_every
        Read out spins and record the best energy every this many steps, following the
        protocol of Kanao and Goto, who evaluate periodically rather than only at the
        end.
    """

    n_steps: int = 1000
    dt: float = 0.7
    c1: float = 0.6
    a0: float = 1.0
    n_replicas: int = 32
    evaluate_every: int = 20


def derive_coupling_strength(couplings: NDArray[np.float64], c1: float) -> float:
    """Derive ``c_0`` from the coupling matrix.

    Kanao and Goto give ``c_0 = c_1 / sqrt(N)`` (their Eq. 12) for the
    Sherrington-Kirkpatrick model, whose couplings are drawn from ``{-1, +1}`` and
    therefore have unit standard deviation. The folding Hamiltonian's couplings are
    nothing like unit scale -- the derived penalty weights make them span orders of
    magnitude -- so the constant has to be normalised by the actual spread::

        c_0 = c_1 / (sigma * sqrt(N))

    with ``sigma`` the standard deviation of the off-diagonal couplings. This reduces to
    the published formula exactly when ``sigma == 1``, which is the case it was tuned
    for. Without the normalisation, the coupling term would swamp the bifurcation term
    by a factor of thousands and every trajectory would slam into the walls immediately.
    """
    size = couplings.shape[0]
    off_diagonal = couplings[np.triu_indices(size, k=1)]
    spread = float(np.std(off_diagonal)) if off_diagonal.size else 0.0
    if spread <= 0.0:
        # No couplings at all: the problem is separable and c_0 is irrelevant, but it
        # must stay finite.
        spread = 1.0
    return float(c1 / (spread * np.sqrt(max(size, 1))))


def ballistic_bifurcation(
    ising: IsingModel,
    seed: int,
    config: BifurcationConfig | None = None,
) -> tuple[float, NDArray[np.int_], int]:
    """Integrate ballistic SB over a batch of replicas and return the best state.

    Returns the best Ising energy found, the spin vector achieving it, and the number
    of energy evaluations performed. Fully determined by ``seed``.
    """
    config = config or BifurcationConfig()
    rng = np.random.default_rng(seed)

    size = ising.n_variables
    # The stored coupling matrix is strictly upper triangular; the force needs the
    # symmetric form so that each spin feels every partner exactly once.
    symmetric = ising.couplings + ising.couplings.T
    fields = ising.fields

    c0 = derive_coupling_strength(symmetric, config.c1)

    # Positions and momenta start small and random, as in the reference implementation.
    x = rng.uniform(-0.1, 0.1, size=(config.n_replicas, size))
    y = rng.uniform(-0.1, 0.1, size=(config.n_replicas, size))

    best_energy = np.inf
    best_spins = np.sign(x[0])
    evaluations = 0

    for step in range(config.n_steps):
        # Control parameter ramped linearly from 0 to a0 (Kanao and Goto, Sec. II).
        a = config.a0 * step / max(config.n_steps - 1, 1)

        # The force is minus the gradient of the Ising energy stored by this package,
        # E = sum_{i<j} J_ij s_i s_j + sum_i h_i s_i, which carries the opposite sign
        # convention to the reference papers' E = -(1/2) sum J_ij s_i s_j.
        force = -(x @ symmetric + fields)

        # Symplectic Euler: momentum first, then position using the updated momentum.
        y += config.dt * (-(config.a0 - a) * x + c0 * force)
        x += config.dt * config.a0 * y

        # Perfectly inelastic walls at x = +/-1: this is what makes the method
        # *ballistic* rather than adiabatic.
        outside = np.abs(x) > 1.0
        if outside.any():
            x[outside] = np.sign(x[outside])
            y[outside] = 0.0

        if step % config.evaluate_every == 0 or step == config.n_steps - 1:
            spins = np.sign(x)
            spins[spins == 0.0] = 1.0
            energies = (
                np.einsum("ri,ij,rj->r", spins, symmetric, spins) / 2.0
                + spins @ fields
                + ising.offset
            )
            evaluations += config.n_replicas
            index = int(np.argmin(energies))
            if energies[index] < best_energy:
                best_energy = float(energies[index])
                best_spins = spins[index].copy()

    return best_energy, best_spins.astype(np.int64), evaluations


class BifurcationSolver:
    """Ballistic simulated bifurcation, as a :class:`~foldq.solvers.base.Solver`."""

    def __init__(self, config: BifurcationConfig | None = None) -> None:
        """Store the run configuration."""
        self._config = config or BifurcationConfig()

    @property
    def name(self) -> str:
        """Short identifier used in artifacts, tables and figures."""
        return "bifurcation"

    def solve(
        self, hamiltonian: FoldingHamiltonian, seed: int | None = None
    ) -> SolverResult:
        """Integrate the Hamiltonian's Ising form and decode the result."""
        if seed is None:
            message = "simulated bifurcation requires an explicit seed"
            raise ValueError(message)

        ising = hamiltonian.to_ising()
        start = time.perf_counter()
        _, spins, evaluations = ballistic_bifurcation(ising, seed, self._config)
        bits = hamiltonian.repair_auxiliaries(((1 - spins) // 2).astype(np.int64))
        energy = hamiltonian.to_qubo().energy(bits)
        elapsed = time.perf_counter() - start

        turns = hamiltonian.decode_turns(bits)
        return SolverResult(
            energy=energy,
            turns=turns,
            n_evaluations=evaluations,
            wall_time_s=elapsed,
            seed=seed,
            metadata={
                "solver": self.name,
                "n_variables": ising.n_variables,
                "n_steps": self._config.n_steps,
                "n_replicas": self._config.n_replicas,
                "dt": self._config.dt,
                "c1": self._config.c1,
            },
        )
