"""Render RESULTS.md tables directly from a benchmark artifact.

The tables in RESULTS.md are generated, never typed. That is the mechanical enforcement
of Guardrail 1: a number cannot appear in the write-up unless it is in a committed
artifact, because there is no path by which a human types one in.
"""

from __future__ import annotations

from typing import Any


def _format(value: float | None, spec: str = ".2f", missing: str = "n/a") -> str:
    """Format a number, or a placeholder where the quantity is undefined."""
    if value is None:
        return missing
    return format(value, spec)


def instance_table(report: dict[str, Any]) -> str:
    """Return the per-instance resource table."""
    rows = sorted(
        report["instances"], key=lambda r: (r["n_beads"], r["sequence"], r["encoding"])
    )
    header = (
        "| sequence | N | encoding | locality | turn | contact "
        "| auxiliary | total | exact optimum |"
    )
    lines = [header, "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(
            f"| `{row['sequence']}` | {row['n_beads']} | {row['encoding']} "
            f"| {row['core_degree']} | {row['n_turn_qubits']} "
            f"| {row['n_contact_qubits']} | {row['n_auxiliary_qubits']} "
            f"| {row['total_qubits']} | {row['optimum']:.0f} |"
        )
    return "\n".join(lines)


def solver_table(report: dict[str, Any]) -> str:
    """Return the per-solver results table."""
    rows = sorted(
        report["summaries"],
        key=lambda r: (r["n_beads"], r["encoding"], r["solver"]),
    )
    header = (
        "| N | encoding | solver | p_s | best E | optimum "
        "| approx. ratio | mean run (s) | TTS @99% (s) |"
    )
    lines = [header, "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(
            f"| {row['n_beads']} | {row['encoding']} | {row['solver']} "
            f"| {row['success_probability']:.2f} | {row['best_energy']:.0f} "
            f"| {row['optimum']:.0f} | {_format(row['approximation_ratio'])} "
            f"| {row['mean_wall_time_s']:.3f} "
            f"| {_format(row['time_to_solution_s'], '.2f', 'not reached')} |"
        )
    return "\n".join(lines)


def provenance(report: dict[str, Any]) -> str:
    """Return a short provenance block for the artifact."""
    env = report["environment"]
    config = report["config"]
    return "\n".join(
        [
            f"- profile: `{config['label']}`, {len(config['sequences'])} sequences "
            f"x {len(config['encodings'])} encodings x {config['n_seeds']} seeds",
            f"- commit: `{env['git_sha']}`",
            f"- generated: {env['timestamp_utc']}",
            f"- python {env['python']} on {env['platform']}",
            f"- numpy {env['version_numpy']}, scipy {env['version_scipy']}, "
            f"qiskit {env['version_qiskit']}",
        ]
    )


def render(report: dict[str, Any]) -> str:
    """Return the generated section of RESULTS.md for one artifact."""
    return "\n\n".join(
        [
            "### Provenance",
            provenance(report),
            "### Instances and encoding cost",
            instance_table(report),
            "### Solver results",
            solver_table(report),
        ]
    )
