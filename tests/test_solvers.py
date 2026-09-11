"""Tests for the classical and quantum-inspired solvers.

Every solver is handed the *same* Hamiltonian object, so any difference in what they
find is a difference in the algorithm rather than in the problem.

The most important test here is not a success rate. It is
``test_core_minimum_equals_the_physical_optimum``: a Hamiltonian whose global minimum
sits below the true ground state will happily report an excellent energy for an
unphysical bit pattern, and no amount of solver benchmarking would reveal it. That test
caught exactly such a bug in the one-hot validity penalty.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from foldq.encoding import DENSE, ONE_HOT, TurnEncoding
from foldq.hamiltonian import build_hamiltonian
from foldq.lattice import TetrahedralLattice
from foldq.peptide import FoldingInstance, Peptide
from foldq.solvers.annealing import AnnealingConfig, AnnealingSolver
from foldq.solvers.bifurcation import (
    BifurcationConfig,
    BifurcationSolver,
    derive_coupling_strength,
)
from foldq.solvers.exact import exhaustive_search

LATTICE = TetrahedralLattice()
ENCODINGS = [DENSE, ONE_HOT]

#: Annealing configuration used for the success-rate tests. Auxiliaries are slaved to
#: the primaries, which is the move set that makes a quadratized QUBO tractable.
SLAVED = AnnealingConfig(n_sweeps=800, slave_auxiliaries=True)


def _instance(sequence: str) -> FoldingInstance:
    """Build a tetrahedral folding instance for an HP sequence."""
    return FoldingInstance(Peptide(sequence), LATTICE)


# ---------------------------------------------------------------------------
# The Hamiltonian must not reward unphysical states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
@pytest.mark.parametrize("sequence", ["HPHPPH", "HHPPHH"])
def test_core_minimum_equals_the_physical_optimum(
    encoding: TurnEncoding, sequence: str
) -> None:
    """The Hamiltonian's global minimum over *all* bit patterns is the true optimum.

    Not merely over the patterns that decode to valid folds. If any unphysical
    assignment scored lower, every solver would be rewarded for finding it, and the
    benchmark would be measuring the wrong thing entirely.

    This is a regression test. Deriving the one-hot validity weight from the contact
    energy scale rather than from the penalty terms it has to dominate produced a
    Hamiltonian whose minimum was -88 against a true optimum of -1, because breaking
    the one-hot constraint let the much larger ``lambda_1`` terms go negative.
    """
    instance = _instance(sequence)
    hamiltonian = build_hamiltonian(instance, encoding=encoding)
    n_primary = hamiltonian.layout.n_primary_qubits

    minimum = min(
        hamiltonian.core.evaluate(dict(enumerate(bits)))
        for bits in itertools.product((0, 1), repeat=n_primary)
    )
    assert minimum == pytest.approx(exhaustive_search(instance).energy)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_both_encodings_agree_on_the_ground_state(encoding: TurnEncoding) -> None:
    """Dense and one-hot describe the same physics, whatever their qubit counts."""
    instance = _instance("HHPHPPHH")
    energy, turns, _ = build_hamiltonian(instance, encoding=encoding).ground_state()
    assert energy == pytest.approx(exhaustive_search(instance).energy)
    assert LATTICE.is_self_avoiding(turns)


def test_one_hot_trades_qubits_for_locality() -> None:
    """One-hot uses more turn qubits but yields a lower-degree Hamiltonian.

    The central resource trade-off of the project, asserted rather than described: the
    dense encoding is 5-local, the one-hot encoding 3-local, and the reduction in degree
    is what makes the quadratized form tractable for local solvers.
    """
    instance = _instance("HHPHPPHH")
    dense = build_hamiltonian(instance, encoding=DENSE)
    one_hot = build_hamiltonian(instance, encoding=ONE_HOT)

    assert one_hot.layout.n_turn_qubits == 2 * dense.layout.n_turn_qubits
    assert dense.core_degree == 5
    assert one_hot.core_degree == 3
    assert one_hot.layout.n_auxiliary_qubits < dense.layout.n_auxiliary_qubits


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_annealing_is_reproducible(encoding: TurnEncoding) -> None:
    """The same seed gives bit-for-bit identical output (Guardrail 4)."""
    hamiltonian = build_hamiltonian(_instance("HPHPPH"), encoding=encoding)
    solver = AnnealingSolver(AnnealingConfig(n_sweeps=100))
    first = solver.solve(hamiltonian, seed=7)
    second = solver.solve(hamiltonian, seed=7)
    assert first.energy == second.energy
    assert first.turns == second.turns
    assert first.n_evaluations == second.n_evaluations


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: e.name)
def test_bifurcation_is_reproducible(encoding: TurnEncoding) -> None:
    """The same seed gives bit-for-bit identical output (Guardrail 4)."""
    hamiltonian = build_hamiltonian(_instance("HPHPPH"), encoding=encoding)
    solver = BifurcationSolver(BifurcationConfig(n_steps=200, n_replicas=8))
    first = solver.solve(hamiltonian, seed=3)
    second = solver.solve(hamiltonian, seed=3)
    assert first.energy == second.energy
    assert first.turns == second.turns


def test_different_seeds_explore_differently() -> None:
    """Seeding actually varies the trajectory, so the seed is not being ignored."""
    hamiltonian = build_hamiltonian(_instance("HHPHPPHH"), encoding=ONE_HOT)
    solver = BifurcationSolver(BifurcationConfig(n_steps=200, n_replicas=4))
    energies = {solver.solve(hamiltonian, seed=s).energy for s in range(8)}
    assert len(energies) > 1


@pytest.mark.parametrize(
    "solver", [AnnealingSolver(), BifurcationSolver()], ids=["annealing", "bifurcation"]
)
def test_stochastic_solvers_require_an_explicit_seed(solver: object) -> None:
    """Omitting the seed is an error, not a silently random run."""
    hamiltonian = build_hamiltonian(_instance("HPHPPH"))
    with pytest.raises(ValueError, match="seed"):
        solver.solve(hamiltonian, seed=None)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Solution quality
# ---------------------------------------------------------------------------


def _success_rate(
    sequence: str, encoding: TurnEncoding, n_seeds: int = 50, sweeps: int = 800
) -> float:
    """Return the fraction of seeded runs that reach the exact optimum."""
    instance = _instance(sequence)
    optimum = exhaustive_search(instance).energy
    hamiltonian = build_hamiltonian(instance, encoding=encoding)
    solver = AnnealingSolver(AnnealingConfig(n_sweeps=sweeps, slave_auxiliaries=True))
    hits = sum(
        solver.solve(hamiltonian, seed=seed).energy <= optimum + 1e-9
        for seed in range(n_seeds)
    )
    return hits / n_seeds


@pytest.mark.parametrize(
    ("sequence", "encoding"),
    [("HPHPPH", DENSE), ("HPHPPH", ONE_HOT), ("HHPHPPHH", DENSE)],
    ids=["N6-dense", "N6-one_hot", "N8-dense"],
)
@pytest.mark.slow
def test_annealing_reaches_the_optimum_in_at_least_95_percent_of_runs(
    sequence: str, encoding: TurnEncoding
) -> None:
    """M3's acceptance criterion, over 50 seeded runs.

    Uses the auxiliary-slaved move set. Annealing that treats every variable as
    independent fails badly on the quadratized QUBO -- 0.05 at N=8 -- which is a finding
    about the encoding rather than about annealing; see RESULTS.md.
    """
    assert _success_rate(sequence, encoding) >= 0.95


@pytest.mark.slow
def test_one_hot_at_n8_is_recorded_rather_than_claimed() -> None:
    """One-hot at N=8 reaches the optimum in roughly four runs in five, not 19 in 20.

    It does *not* meet the 95% bar, and this test records the measured behaviour
    instead of pretending otherwise. The reason is a genuine trade-off rather than a
    defect: slaving auxiliaries makes the effective search space the primary variables,
    and one-hot has 24 of those against dense's 14. So the encoding that is easier for a
    solver working on the raw QUBO is *harder* for one that exploits the auxiliary
    structure. The two encodings win under opposite move sets.

    The bound is loose on purpose. It is a regression guard against the rate collapsing,
    not a precise claim about a stochastic quantity measured over 50 seeds.
    """
    assert _success_rate("HHPHPPHH", ONE_HOT) >= 0.6


@pytest.mark.slow
def test_bifurcation_reaches_the_optimum_on_the_one_hot_encoding() -> None:
    """Ballistic SB solves the 3-local encoding, where it fails on the 5-local one.

    The comparison that motivates keeping both encodings: the same solver, the same
    instance, the same physics, and a success rate that depends entirely on how the
    conformation was written into qubits.
    """
    instance = _instance("HPHPPH")
    optimum = exhaustive_search(instance).energy
    solver = BifurcationSolver(BifurcationConfig(n_replicas=256))

    one_hot = build_hamiltonian(instance, encoding=ONE_HOT)
    hits = sum(
        solver.solve(one_hot, seed=seed).energy <= optimum + 1e-9 for seed in range(20)
    )
    assert hits >= 15, f"one-hot: reached the optimum in {hits}/20 runs"


def test_solutions_decode_to_self_avoiding_conformations() -> None:
    """When a solver finds the optimum, the decoded fold is physically valid."""
    instance = _instance("HPHPPH")
    optimum = exhaustive_search(instance).energy
    hamiltonian = build_hamiltonian(instance, encoding=ONE_HOT)
    result = AnnealingSolver(SLAVED).solve(hamiltonian, seed=0)
    assert result.energy == pytest.approx(optimum)
    assert result.turns is not None
    assert LATTICE.is_self_avoiding(result.turns)


# ---------------------------------------------------------------------------
# Simulated bifurcation specifics
# ---------------------------------------------------------------------------


def test_coupling_strength_normalises_by_the_coupling_spread() -> None:
    """``c_0`` scales as ``1 / (sigma sqrt(N))``, reducing to Eq. 12 when sigma is 1.

    Without the ``sigma`` normalisation the published constant, tuned for couplings
    drawn from {-1, +1}, would be wrong by the orders of magnitude that this
    Hamiltonian's derived penalty weights span.
    """
    size = 16
    rng = np.random.default_rng(0)
    unit = rng.choice([-1.0, 1.0], size=(size, size)).astype(np.float64)
    unit = np.triu(unit, 1)
    unit = unit + unit.T

    c1 = 0.6
    base = derive_coupling_strength(unit, c1)
    scaled = derive_coupling_strength((1000.0 * unit).astype(np.float64), c1)

    assert base == pytest.approx(c1 / np.sqrt(size), rel=0.2)
    assert scaled == pytest.approx(base / 1000.0, rel=1e-6)


@pytest.mark.slow
def test_bifurcation_is_vectorised_over_replicas() -> None:
    """A batch of replicas costs far less per sample than running them one at a time.

    This is the property that makes simulated bifurcation quantum-*inspired* in a way
    that matters for hardware: every replica shares one matrix product per step, so the
    algorithm maps onto wide parallel silicon. Simulated annealing has no equivalent,
    because each accepted flip changes the fields the next proposal depends on.
    """
    hamiltonian = build_hamiltonian(_instance("HHPHPPHH"), encoding=ONE_HOT)
    batch = 64

    single = BifurcationSolver(BifurcationConfig(n_steps=300, n_replicas=1))
    start = time.perf_counter()
    for seed in range(batch):
        single.solve(hamiltonian, seed=seed)
    sequential = time.perf_counter() - start

    batched = BifurcationSolver(BifurcationConfig(n_steps=300, n_replicas=batch))
    start = time.perf_counter()
    batched.solve(hamiltonian, seed=0)
    vectorised = time.perf_counter() - start

    assert vectorised < sequential / 4, (
        f"batched {batch} replicas took {vectorised:.3f}s against "
        f"{sequential:.3f}s run singly; expected at least a 4x saving"
    )
