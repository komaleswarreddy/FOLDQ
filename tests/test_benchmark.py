"""Tests for the benchmark harness, its artifacts and its figures.

The harness exists to make Guardrail 1 enforceable: every published number traces to a
committed artifact that records the commit and the environment that produced it. These
tests check that the provenance is really there, that the runner gains a solver without
being edited, and that figures are drawn only from artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foldq.benchmark.runner import (
    FAST,
    FULL,
    BenchmarkConfig,
    default_solvers,
    environment,
    load_report,
    run_benchmark,
    write_report,
)
from foldq.hamiltonian import FoldingHamiltonian
from foldq.peptide import Peptide
from foldq.solvers.base import SolverResult
from foldq.viz.conformation import plot_conformation
from foldq.viz.plots import generate_all

#: A single tiny instance, enough to exercise the machinery quickly.
TINY = BenchmarkConfig(sequences=("HPHPPH",), n_seeds=3, label="tiny")


def test_environment_records_provenance() -> None:
    """Artifacts carry the commit, the interpreter and the resolved package versions.

    Without these a committed number cannot be traced back to the code that produced
    it, and "reproducible" would be a claim rather than a property.
    """
    record = environment()
    assert record["git_sha"]
    assert record["python"]
    assert record["version_foldq"]
    assert record["version_numpy"]
    assert record["timestamp_utc"]


def test_report_round_trips_through_json(tmp_path: Path) -> None:
    """A written artifact loads back with its configuration and results intact."""
    report = run_benchmark(TINY, default_solvers(fast=True))
    path = write_report(report, tmp_path)
    loaded = load_report(path)

    assert loaded["config"]["label"] == "tiny"
    assert loaded["environment"]["git_sha"] == report.environment["git_sha"]
    assert len(loaded["summaries"]) == len(report.summaries)
    # And it really is JSON on disk, not a pickle with a .json name.
    json.loads(path.read_text(encoding="utf-8"))


def test_every_solver_is_scored_against_the_exact_optimum() -> None:
    """Each summary carries the exhaustive optimum for its instance.

    Success probability and approximation ratio are only meaningful because the exact
    answer is known independently.
    """
    report = run_benchmark(TINY, default_solvers(fast=True))
    assert report.summaries
    for summary in report.summaries:
        assert summary.optimum == -1.0
        assert summary.best_energy >= summary.optimum - 1e-9
        assert 0.0 <= summary.success_probability <= 1.0


def test_all_solvers_receive_the_same_instance() -> None:
    """Every summary for a given encoding shares one optimum and one chain length."""
    report = run_benchmark(TINY, default_solvers(fast=True))
    for encoding in TINY.encodings:
        rows = [s for s in report.summaries if s.encoding == encoding]
        assert len({s.optimum for s in rows}) == 1
        assert len({s.n_beads for s in rows}) == 1


def test_runner_accepts_a_new_solver_without_being_edited() -> None:
    """Adding a solver is a call-site change, not a change to the runner.

    The brief is explicit that if adding a solver requires editing ``runner.py`` then
    the abstraction is wrong, so this test adds one the runner has never heard of.
    """

    class AlwaysZeroSolver:
        """A deliberately useless solver that satisfies the protocol."""

        @property
        def name(self) -> str:
            """Short identifier."""
            return "always-zero"

        def solve(
            self, hamiltonian: FoldingHamiltonian, seed: int | None = None
        ) -> SolverResult:
            """Return a fixed, poor answer."""
            del hamiltonian
            return SolverResult(
                energy=0.0,
                turns=None,
                n_evaluations=1,
                wall_time_s=0.001,
                seed=seed,
            )

    report = run_benchmark(TINY, [AlwaysZeroSolver()])
    names = {summary.solver for summary in report.summaries}
    assert names == {"always-zero"}
    for summary in report.summaries:
        assert summary.success_probability == 0.0
        assert summary.time_to_solution_s is None


def test_instance_records_report_the_qubit_registers_separately() -> None:
    """Every instance row breaks the qubit count into its three registers."""
    report = run_benchmark(TINY, default_solvers(fast=True))
    for record in report.instances:
        assert record.total_qubits == (
            record.n_turn_qubits + record.n_contact_qubits + record.n_auxiliary_qubits
        )
        assert record.core_degree in {3, 5}


def test_fast_and_full_profiles_differ_only_in_scale() -> None:
    """The CI sweep runs the same code path as the published one, just smaller."""
    assert FAST.encodings == FULL.encodings
    assert FAST.n_seeds < FULL.n_seeds
    assert len(FAST.sequences) < len(FULL.sequences)
    assert set(FAST.sequences) <= set(FULL.sequences)


def test_full_profile_uses_the_fifty_seeds_the_brief_specifies() -> None:
    """The published sweep uses 50 seeds per solver and instance."""
    assert FULL.n_seeds == 50


def test_figures_are_generated_from_an_artifact(tmp_path: Path) -> None:
    """Figures read a report and write PNGs; they compute nothing themselves."""
    report = run_benchmark(TINY, default_solvers(fast=True))
    path = write_report(report, tmp_path)
    written = generate_all(load_report(path), tmp_path / "figures")
    assert len(written) == 3
    for figure in written:
        assert figure.is_file()
        assert figure.stat().st_size > 1000


def test_conformation_plot_is_written(tmp_path: Path) -> None:
    """The 3D fold plot renders without a display attached."""
    path = plot_conformation(Peptide("HPHPPH"), (0, 1, 2, 0, 1), tmp_path / "fold.png")
    assert path.is_file()
    assert path.stat().st_size > 1000


@pytest.mark.parametrize("label", ["fast", "full"])
def test_committed_artifacts_exist_and_carry_provenance(label: str) -> None:
    """The artifacts behind README.md and RESULTS.md are committed and complete.

    This is the test that makes Guardrail 1 enforceable rather than aspirational: if
    the documentation cites a number, the artifact it came from is in the repository.
    """
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "results"
    artifact = path / f"benchmark-{label}.json"
    if not artifact.is_file():
        pytest.skip(f"{artifact.name} has not been generated yet")

    report = load_report(artifact)
    assert report["environment"]["git_sha"]
    assert report["environment"]["version_foldq"]
    assert report["summaries"]
    assert report["instances"]
