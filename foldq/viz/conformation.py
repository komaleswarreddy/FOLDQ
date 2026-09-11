"""Three-dimensional plot of a folded chain."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from foldq.lattice import TetrahedralLattice
from foldq.peptide import Peptide

#: Hydrophobic beads are drawn dark and polar ones light, because the whole point of
#: the HP model is whether the hydrophobic residues end up buried together.
_HYDROPHOBIC_COLOUR = "#1b3a5c"
_POLAR_COLOUR = "#c8d6e2"


def plot_conformation(
    peptide: Peptide,
    turns: Sequence[int],
    path: Path,
    title: str | None = None,
) -> Path:
    """Draw a folded chain in three dimensions and save it.

    Bonds are drawn along the chain and the scored hydrophobic contacts as dashed red
    lines, so the contacts the energy actually counts are visible rather than inferred
    from the geometry by eye.
    """
    lattice = TetrahedralLattice()
    positions = lattice.walk(turns)

    figure = plt.figure(figsize=(6, 5))
    axes = figure.add_subplot(111, projection="3d")
    axes.plot(
        [p[0] for p in positions],
        [p[1] for p in positions],
        [p[2] for p in positions],
        color="#555555",
        linewidth=1.5,
    )

    for index, (x, y, z) in enumerate(positions):
        hydrophobic = peptide.is_hydrophobic(index)
        axes.scatter(
            [x],
            [y],
            [z],
            s=160,
            color=_HYDROPHOBIC_COLOUR if hydrophobic else _POLAR_COLOUR,
            edgecolors="#222222",
            depthshade=False,
        )
        axes.text(x, y, z, f" {index}", fontsize=7, color="#333333")

    for i, j in lattice.contact_pairs(positions):
        if peptide.is_hydrophobic(i) and peptide.is_hydrophobic(j):
            axes.plot(
                [positions[i][0], positions[j][0]],
                [positions[i][1], positions[j][1]],
                [positions[i][2], positions[j][2]],
                color="#b03030",
                linestyle="--",
                linewidth=1.2,
            )

    axes.set_title(title or f"{peptide.sequence} on the tetrahedral lattice")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    axes.set_box_aspect((1, 1, 1))

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
