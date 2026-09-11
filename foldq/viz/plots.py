"""Figures generated from a committed benchmark artifact.

Every figure here reads a JSON report written by :mod:`foldq.benchmark.runner`. None of
them computes anything, so a figure can never show a number that is not also in a
committed artifact.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

#: One colour per solver, kept consistent across every figure.
_COLOURS = {
    "annealing": "#d1731f",
    "annealing-slaved": "#2b6ca3",
    "bifurcation": "#3f8f4f",
}

#: Solid for the dense encoding, dashed for one-hot.
_STYLES = {"dense": "-", "one_hot": "--"}


def _series(
    summaries: list[dict[str, Any]], field: str
) -> dict[tuple[str, str], list[tuple[int, float]]]:
    """Group one field by solver and encoding, sorted by chain length."""
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in summaries:
        value = row[field]
        if value is None:
            continue
        grouped[(row["solver"], row["encoding"])].append((row["n_beads"], value))
    return {key: sorted(points) for key, points in grouped.items()}


def _save(figure: Any, path: Path) -> Path:
    """Save and close a figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_success_probability(report: dict[str, Any], path: Path) -> Path:
    """Plot success probability against chain length for every solver and encoding."""
    figure, axes = plt.subplots(figsize=(7.5, 4.5))
    for (solver, encoding), points in _series(
        report["summaries"], "success_probability"
    ).items():
        axes.plot(
            [n for n, _ in points],
            [v for _, v in points],
            marker="o",
            color=_COLOURS.get(solver, "#666666"),
            linestyle=_STYLES.get(encoding, "-"),
            label=f"{solver} ({encoding})",
        )
    axes.axhline(0.95, color="#999999", linewidth=0.8, linestyle=":")
    axes.set_xlabel("chain length N (beads)")
    axes.set_ylabel("success probability")
    axes.set_ylim(-0.05, 1.05)
    axes.set_title("Reaching the exact optimum (dotted line: 95% target)")
    axes.legend(fontsize=7)
    return _save(figure, path)


def plot_time_to_solution(report: dict[str, Any], path: Path) -> Path:
    """Plot time-to-solution at 99% confidence.

    A solver that never reached the optimum has no finite time-to-solution and is
    simply absent from its curve, rather than being drawn at some large value that
    would read as a measurement.
    """
    figure, axes = plt.subplots(figsize=(7.5, 4.5))
    for (solver, encoding), points in _series(
        report["summaries"], "time_to_solution_s"
    ).items():
        axes.plot(
            [n for n, _ in points],
            [v for _, v in points],
            marker="o",
            color=_COLOURS.get(solver, "#666666"),
            linestyle=_STYLES.get(encoding, "-"),
            label=f"{solver} ({encoding})",
        )
    axes.set_yscale("log")
    axes.set_xlabel("chain length N (beads)")
    axes.set_ylabel("time to solution at 99% confidence (s)")
    axes.set_title("Time to solution (missing points never reached the optimum)")
    axes.legend(fontsize=7)
    return _save(figure, path)


def plot_qubit_scaling(report: dict[str, Any], path: Path) -> Path:
    """Plot how each qubit register grows with chain length, per encoding."""
    figure, axes = plt.subplots(figsize=(7.5, 4.5))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in report["instances"]:
        grouped[record["encoding"]].append(record)

    registers = [
        ("n_turn_qubits", "turn", "#2b6ca3"),
        ("n_contact_qubits", "contact", "#d1731f"),
        ("total_qubits", "total", "#3f8f4f"),
    ]
    for encoding, records in grouped.items():
        ordered = sorted(records, key=lambda record: record["n_beads"])
        for field, label, colour in registers:
            axes.plot(
                [record["n_beads"] for record in ordered],
                [record[field] for record in ordered],
                marker="o",
                color=colour,
                linestyle=_STYLES.get(encoding, "-"),
                label=f"{label} ({encoding})",
            )
    axes.set_yscale("log")
    axes.set_xlabel("chain length N (beads)")
    axes.set_ylabel("variables (log scale)")
    axes.set_title("Qubit cost by register: solid dense, dashed one-hot")
    axes.legend(fontsize=7, ncol=2)
    return _save(figure, path)


def generate_all(report: dict[str, Any], directory: Path) -> list[Path]:
    """Generate every figure from one report."""
    return [
        plot_success_probability(report, directory / "success-probability.png"),
        plot_time_to_solution(report, directory / "time-to-solution.png"),
        plot_qubit_scaling(report, directory / "qubit-scaling.png"),
    ]
