# FoldQ

> **For peptide folding at today's hardware scale, how do variational quantum algorithms
> actually compare against classical and quantum-inspired classical solvers on the
> identical problem instance?**

FoldQ is a reproducible benchmark, not a demonstration. It encodes a coarse-grained
peptide on a tetrahedral lattice as a single optimisation instance, hands that *same*
instance to several solvers, and reports which one wins under which conditions —
including the cases where the quantum approach loses.

The full specification, including the physics, the milestone plan and the guardrails
this repository is built under, is in [PROJECT_BRIEF.md](PROJECT_BRIEF.md).

---

## Results

Full write-up and verdict: **[RESULTS.md](RESULTS.md)**.

From the committed `benchmark-full.json` artifact — 6 sequences x 2 encodings x 3 solvers
x 50 seeds, about 55 minutes of compute:

| N | encoding | naive annealing | slaved annealing | ballistic SB |
| --- | --- | --- | --- | --- |
| 6 | dense | 0.86 | **1.00** | 0.00 |
| 6 | one-hot | 0.72 | **1.00** | **0.90** |
| 8 | dense | 0.02 | **1.00** | 0.00 |
| 8 | one-hot | 0.04 | 0.82 | 0.00 |
| 10 | dense | 0.00 | 0.14 | 0.00 |
| 10 | one-hot | 0.00 | 0.14 | 0.00 |

Success probability over 50 seeded runs, against the exact optimum from exhaustive
enumeration.

**The quantum-inspired solver loses.** Ballistic simulated bifurcation reaches the exact
optimum on one instance class in the whole sweep — six beads under the one-hot encoding —
and scores 0.00 everywhere else; from N=7 up, the best energy it finds is *positive*,
meaning an infeasible conformation. Simulated annealing with a structure-aware move set
holds 1.00 to N=9.

The one place it works is the second finding: at N=6 the same solver on the same instance
scores **0.00 on the dense encoding and 0.90 on the one-hot encoding**. The dense encoding
minimises qubit count, and reaching a quadratic form from it introduces penalty weights
~10^4 above the physical energy scale, so a contact worth -1 becomes a rounding error.

**No variational quantum solver was benchmarked**, so the quantum half of the headline
question is unanswered. See RESULTS.md for the full limitations.

---

## Status

This build targets a deliberately reduced scope: the lattice model, a validated
Hamiltonian, the classical and quantum-inspired solvers, and the benchmark harness. The
variational quantum solver, the noise study, the HTTP service and the viewer are **not
implemented**, and this README will not claim otherwise until they are.

| Milestone | Scope | Status |
| --- | --- | --- |
| M0 — Scaffold | Package, tooling, CI | **Done** |
| M1 — Lattice and exhaustive baseline | Tetrahedral geometry, self-avoidance, HP contacts | **Done** |
| M2 — Hamiltonian construction | Pauli / QUBO / Ising views, derived penalty weights | **Done** |
| M3 — Classical and quantum-inspired solvers | Simulated annealing, simulated bifurcation | **Done** |
| M4 — Variational quantum solver | VQE / QAOA with CVaR | Out of scope for this build |
| M5 — Noise and mitigation study | Depolarising / thermal sweeps, M3 and ZNE | Out of scope for this build |
| M6 — Benchmark harness and results | Multi-seed sweeps, artifacts, figures, RESULTS.md | **Done** |
| M7 — Service layer | FastAPI job submission, Docker | Out of scope for this build |
| M8 — Viewer and writeup | Streamlit artifact viewer | Out of scope for this build |

Optional dependency groups for the out-of-scope milestones (`[quantum]`, `[reference]`)
are declared in `pyproject.toml` so the boundary is explicit rather than implied, but
nothing in the package imports them yet.

## Validation so far

The lattice geometry and the self-avoidance predicate are checked against published
enumerations that nothing in this repository chose:

| Lattice | Self-avoiding walks, 0–9 steps | Source |
| --- | --- | --- |
| Simple cubic | 1, 6, 30, 150, 726, 3534, 16926, 81390, 387966, 1853886 | [OEIS A001412](https://oeis.org/A001412) |
| Diamond (tetrahedral) | 1, 4, 12, 36, 108, 324, 948, 2796, 8196, 24060 | [A227715](https://oeis.org/A227715) + [A227716](https://oeis.org/A227716), summed over endpoint |

The cubic lattice is a validation geometry only — the qubit encoding never uses it. It
is here because the published HP benchmark sequences begin at N ≈ 20, far beyond
exhaustive enumeration on any lattice, so walk counts rather than HP ground states are
what provides a genuine external check at sizes that can be enumerated exactly.

### Solvers: the encoding decides the outcome

Both encodings describe identical physics and agree on the ground state. They do not
behave alike under a solver, and the ranking depends on the move set. The measured table
is in [RESULTS.md](RESULTS.md) and above; the mechanism is:

**On the raw QUBO, one-hot wins.** Rosenberg degree reduction of the 5-local dense
encoding produces penalty weights ~10^4 above the physical energy scale, and a contact
worth -1 becomes a rounding error on that landscape. Ballistic SB never resolves it. The
3-local one-hot encoding needs far less reduction, and bSB starts working.

**Under a structure-aware move set, dense wins.** Auxiliaries are *determined* by the
primary variables, so slaving them to their defining products shrinks the effective
search to the primaries -- 14 for dense against 24 for one-hot. The encoding friendlier
to a naive solver is the harder one for a solver that exploits the structure.

Qubit efficiency and solver-friendliness are not the same axis, and the dense encoding
that minimises qubit count is the one a near-term Ising machine handles worst.

### Hamiltonian resources

The Hamiltonian follows Robert et al., npj Quantum Information **7**, 38 (2021)
([arXiv:1908.02163](https://arxiv.org/abs/1908.02163)), with the dense two-qubit-per-turn
encoding. One pseudo-Boolean polynomial is the single source of truth; the Pauli, QUBO
and Ising views are all exported from it.

| N | mode | turn | contact | auxiliary | total | locality |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | published | 8 | 2 | 22 | 32 | 5 |
| 7 | strict SAW | 8 | 2 | 34 | 44 | 8 |
| 9 | published | 12 | 6 | 84 | 102 | 5 |
| 9 | strict SAW | 12 | 6 | 288 | 306 | 12 |

The turn register is 2(N−3), matching the paper. The published model's **locality stays
at 5 for every chain length** — that is its central resource claim, reproduced here from
our own construction rather than quoted.

Auxiliary qubits come from Rosenberg degree reduction: the dense encoding makes the
distance quartic and the gated interaction term quintic, and a QUBO or Ising `(J, h)` is
quadratic by definition. Reporting only the 2(N−3) turn register would understate the
width by roughly 3×.

`enforce_global_saw=True` adds exact excluded-volume constraints. The published model
protects only against overlaps adjacent to a claimed contact; measured over every
sequence tested, its ground state is nonetheless a valid self-avoiding fold at N ≤ 9,
but that is an empirical result rather than a guarantee. Making it a guarantee costs
locality that grows with N — affordable classically, not on a near-term device.

Symmetry fixing is measured, not assumed: holding the first two turns fixed leaves
exactly 1 in 12 self-avoiding conformations, the order of the tetrahedral rotation
group A₄, and is verified to leave the optimum unchanged.

---

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv will fetch the interpreter
for you).

```bash
git clone https://github.com/komaleswarreddy/FOLDQ.git
cd FOLDQ
uv sync
```

Then, on Linux or macOS:

```bash
make check      # lint, type-check and test — everything CI runs
make test
```

On Windows, where GNU make is not available, use the shim — the target names are
identical, and a test fails the build if the two ever drift apart:

```powershell
.\make.ps1 check
.\make.ps1 test
```

`make help` (or `.\make.ps1 help`) lists every target. `bench`, `bench-fast` and
`figures` are declared but exit non-zero until M6 implements them.

---

## Reproducibility

- `pyproject.toml` declares compatible version ranges; `uv.lock` pins exact, hashed
  versions. CI installs with `uv sync --locked`, which fails rather than re-resolving if
  the lockfile is stale.
- Every stochastic function takes an explicit seed. Two runs with the same seed produce
  identical output.
- Benchmark artifacts will record the git SHA, the resolved package versions and the
  seeds alongside every number.

---

## Reference

The encoding follows Robert, Barkoutsos, Woerner and Tavernelli, *"Resource-efficient
quantum algorithm for protein folding"*, npj Quantum Information **7**, 38 (2021).
Deviations from it will be stated explicitly where they occur.

## Licence

MIT — see [LICENSE](LICENSE).
