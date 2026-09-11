"""Command line interface: ``foldq solve``, ``foldq bench`` and ``foldq figures``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from foldq.benchmark.runner import (
    FAST,
    FULL,
    default_solvers,
    load_report,
    run_benchmark,
    write_report,
)
from foldq.encoding import ENCODINGS
from foldq.hamiltonian import build_hamiltonian
from foldq.lattice import TetrahedralLattice
from foldq.peptide import FoldingInstance, Peptide
from foldq.solvers.exact import exhaustive_search
from foldq.viz.conformation import plot_conformation
from foldq.viz.plots import generate_all

#: Where committed artifacts and figures live.
RESULTS_DIR = Path("benchmarks/results")
FIGURES_DIR = Path("benchmarks/figures")

app = typer.Typer(
    help="A reproducible benchmark for coarse-grained lattice protein folding.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def solve(
    sequence: Annotated[str, typer.Argument(help="HP sequence, e.g. HPHPPH")],
    encoding: Annotated[str, typer.Option(help="dense or one_hot")] = "dense",
    plot: Annotated[bool, typer.Option(help="Write a 3D plot of the fold")] = False,
) -> None:
    """Find the exact ground state of one sequence and report its resources."""
    if encoding not in ENCODINGS:
        message = f"unknown encoding {encoding!r}; choose from {sorted(ENCODINGS)}"
        raise typer.BadParameter(message)

    instance = FoldingInstance(Peptide(sequence), TetrahedralLattice())
    result = exhaustive_search(instance)
    hamiltonian = build_hamiltonian(instance, encoding=ENCODINGS[encoding])
    layout = hamiltonian.layout

    typer.echo(
        f"sequence         {instance.peptide.sequence}  (N = {instance.n_beads})"
    )
    typer.echo(f"exact optimum    {result.energy}")
    typer.echo(f"turns            {result.turns}")
    typer.echo(f"encoding         {encoding}  (locality {hamiltonian.core_degree})")
    typer.echo(
        f"qubits           turn {layout.n_turn_qubits}"
        f" + contact {layout.n_contact_qubits}"
        f" + auxiliary {layout.n_auxiliary_qubits}"
        f" = {layout.total_qubits}"
    )
    typer.echo(f"exact evaluations {result.n_evaluations}")

    if plot and result.turns is not None:
        path = plot_conformation(
            instance.peptide, result.turns, FIGURES_DIR / f"fold-{sequence}.png"
        )
        typer.echo(f"wrote            {path}")


@app.command()
def bench(
    fast: Annotated[bool, typer.Option(help="Run the reduced CI sweep")] = False,
) -> None:
    """Run a benchmark sweep and write a JSON artifact."""
    config = FAST if fast else FULL
    typer.echo(
        f"running the {config.label} sweep: "
        f"{len(config.sequences)} sequences x {len(config.encodings)} encodings "
        f"x {config.n_seeds} seeds"
    )
    report = run_benchmark(config, default_solvers(fast=fast))
    path = write_report(report, RESULTS_DIR)
    typer.echo(f"wrote {path}")
    typer.echo(f"git sha {report.environment['git_sha']}")


@app.command()
def figures(
    label: Annotated[str, typer.Option(help="Which artifact to plot")] = "full",
) -> None:
    """Regenerate every figure from a committed artifact."""
    path = RESULTS_DIR / f"benchmark-{label}.json"
    if not path.is_file():
        message = f"no artifact at {path}; run 'foldq bench' first"
        raise typer.BadParameter(message)
    for written in generate_all(load_report(path), FIGURES_DIR):
        typer.echo(f"wrote {written}")


if __name__ == "__main__":  # pragma: no cover
    app()
