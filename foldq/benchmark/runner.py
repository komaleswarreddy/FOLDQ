"""Benchmark sweeps and reproducible artifact writing.

Every number that appears in README.md or RESULTS.md is produced here and written to a
JSON artifact alongside the git commit, the resolved package versions and the seeds
that generated it. That is what Guardrail 1 requires: a figure with no artifact behind
it does not go in the documentation.

Adding a solver does not require editing this module. The runner is handed a sequence
of objects satisfying :class:`foldq.solvers.base.Solver` and calls each identically.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from foldq.benchmark.metrics import SolverSummary, summarise
from foldq.encoding import ENCODINGS
from foldq.hamiltonian import build_hamiltonian
from foldq.lattice import TetrahedralLattice
from foldq.peptide import FoldingInstance, Peptide
from foldq.solvers.annealing import AnnealingConfig, AnnealingSolver
from foldq.solvers.base import Solver
from foldq.solvers.bifurcation import BifurcationConfig, BifurcationSolver
from foldq.solvers.exact import exhaustive_search

#: Packages whose versions are recorded with every artifact. A benchmark rerun that
#: resolves different versions is a different experiment.
_RECORDED_PACKAGES = ("foldq", "numpy", "scipy", "qiskit")


@dataclass(frozen=True)
class BenchmarkConfig:
    """What to sweep over.

    Attributes
    ----------
    sequences
        HP sequences to fold. Each becomes one instance per encoding.
    encodings
        Names of the turn encodings to compare.
    n_seeds
        Independent seeded runs per (solver, instance). The brief specifies 50; the
        fast profile uses fewer so it can gate CI.
    label
        Profile name, recorded in the artifact and used in its filename.
    """

    sequences: tuple[str, ...]
    encodings: tuple[str, ...] = ("dense", "one_hot")
    n_seeds: int = 50
    label: str = "full"


#: The full sweep. Sequences span N = 6 to 10, which is the range where exhaustive
#: enumeration still provides an exact optimum for every point.
FULL = BenchmarkConfig(
    sequences=(
        "HPHPPH",
        "HHPPHH",
        "HHHHHHH",
        "HHPHPPHH",
        "HPHPPHHPH",
        "HHPPHPPHPH",
    ),
    n_seeds=50,
    label="full",
)

#: A reduced sweep for CI. Same code path, fewer instances and seeds.
FAST = BenchmarkConfig(
    sequences=("HPHPPH", "HHPHPPHH"),
    n_seeds=10,
    label="fast",
)


def default_solvers(fast: bool = False) -> tuple[Solver, ...]:
    """Return the stochastic solvers to benchmark.

    Exhaustive enumeration is not included: it is the ground truth each of these is
    scored against, and it is run separately for every instance.
    """
    sweeps = 200 if fast else 800
    steps = 300 if fast else 1000
    replicas = 32 if fast else 256
    return (
        AnnealingSolver(AnnealingConfig(n_sweeps=sweeps)),
        AnnealingSolver(AnnealingConfig(n_sweeps=sweeps, slave_auxiliaries=True)),
        BifurcationSolver(BifurcationConfig(n_steps=steps, n_replicas=replicas)),
    )


def _solver_label(solver: Solver) -> str:
    """Return a name distinguishing solver variants that share a class."""
    name = solver.name
    config = getattr(solver, "_config", None)
    if name == "annealing" and getattr(config, "slave_auxiliaries", False):
        return "annealing-slaved"
    return name


def environment() -> dict[str, str]:
    """Return the provenance recorded with every artifact.

    Without the commit and the resolved versions, a committed number cannot be traced
    back to the code and environment that produced it, and "reproducible" is a claim
    rather than a property.
    """
    record = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for package in _RECORDED_PACKAGES:
        try:
            record[f"version_{package}"] = metadata.version(package)
        except metadata.PackageNotFoundError:  # pragma: no cover - optional extras
            record[f"version_{package}"] = "not installed"
    return record


def _git_sha() -> str:
    """Return the current commit, marked dirty if the tree has uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


@dataclass(frozen=True)
class InstanceRecord:
    """Exact facts about one instance, independent of any solver."""

    sequence: str
    n_beads: int
    encoding: str
    optimum: float
    exact_evaluations: int
    exact_wall_time_s: float
    n_turn_qubits: int
    n_contact_qubits: int
    n_auxiliary_qubits: int
    total_qubits: int
    core_degree: int


@dataclass(frozen=True)
class BenchmarkReport:
    """A complete sweep: what was run, in what environment, and what happened."""

    config: BenchmarkConfig
    environment: dict[str, str] = field(default_factory=environment)
    instances: list[InstanceRecord] = field(default_factory=list)
    summaries: list[SolverSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view of the whole report."""
        return {
            "config": asdict(self.config),
            "environment": self.environment,
            "instances": [asdict(record) for record in self.instances],
            "summaries": [asdict(summary) for summary in self.summaries],
        }


def run_benchmark(
    config: BenchmarkConfig, solvers: Sequence[Solver] | None = None
) -> BenchmarkReport:
    """Run a sweep and return the report.

    For each sequence and encoding the exact optimum is computed once by exhaustive
    enumeration, then every solver is run ``n_seeds`` times on the *same* Hamiltonian
    object, so no difference between solvers can come from a difference in the problem.
    """
    solvers = solvers if solvers is not None else default_solvers()
    lattice = TetrahedralLattice()
    instances: list[InstanceRecord] = []
    summaries: list[SolverSummary] = []

    for sequence in config.sequences:
        instance = FoldingInstance(Peptide(sequence), lattice)
        exact = exhaustive_search(instance)

        for encoding_name in config.encodings:
            hamiltonian = build_hamiltonian(instance, encoding=ENCODINGS[encoding_name])
            layout = hamiltonian.layout
            instances.append(
                InstanceRecord(
                    sequence=sequence,
                    n_beads=instance.n_beads,
                    encoding=encoding_name,
                    optimum=exact.energy,
                    exact_evaluations=exact.n_evaluations,
                    exact_wall_time_s=exact.wall_time_s,
                    n_turn_qubits=layout.n_turn_qubits,
                    n_contact_qubits=layout.n_contact_qubits,
                    n_auxiliary_qubits=layout.n_auxiliary_qubits,
                    total_qubits=layout.total_qubits,
                    core_degree=hamiltonian.core_degree,
                )
            )

            for solver in solvers:
                results = [
                    solver.solve(hamiltonian, seed=seed)
                    for seed in range(config.n_seeds)
                ]
                summaries.append(
                    summarise(
                        solver=_solver_label(solver),
                        encoding=encoding_name,
                        n_beads=instance.n_beads,
                        optimum=exact.energy,
                        energies=[result.energy for result in results],
                        wall_times=[result.wall_time_s for result in results],
                        evaluations=[result.n_evaluations for result in results],
                    )
                )

    return BenchmarkReport(config=config, instances=instances, summaries=summaries)


def write_report(report: BenchmarkReport, directory: Path) -> Path:
    """Write a report as JSON and return the path written."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"benchmark-{report.config.label}.json"
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_report(path: Path) -> dict[str, Any]:
    """Load a previously written report."""
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded
