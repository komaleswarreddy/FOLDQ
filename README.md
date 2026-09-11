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

**Not yet measured.** No benchmark has been run. This table is the shape the results will
take; every cell will be regenerated from committed artifacts by `make bench`, and no
number will appear here that does not trace to one.

| Solver | Ground-state energy | Approximation ratio | Success probability `p_s` | Time-to-solution |
| --- | --- | --- | --- | --- |
| Exhaustive enumeration (ground truth) | not yet measured | not yet measured | not yet measured | not yet measured |
| Simulated annealing | not yet measured | not yet measured | not yet measured | not yet measured |
| Simulated bifurcation (quantum-inspired) | not yet measured | not yet measured | not yet measured | not yet measured |

---

## Status

This build targets a deliberately reduced scope: the lattice model, a validated
Hamiltonian, the classical and quantum-inspired solvers, and the benchmark harness. The
variational quantum solver, the noise study, the HTTP service and the viewer are **not
implemented**, and this README will not claim otherwise until they are.

| Milestone | Scope | Status |
| --- | --- | --- |
| M0 — Scaffold | Package, tooling, CI | **Done** |
| M1 — Lattice and exhaustive baseline | Tetrahedral geometry, self-avoidance, HP contacts | Planned |
| M2 — Hamiltonian construction | Pauli / QUBO / Ising views, derived penalty weights | Planned |
| M3 — Classical and quantum-inspired solvers | Simulated annealing, simulated bifurcation | Planned |
| M4 — Variational quantum solver | VQE / QAOA with CVaR | Out of scope for this build |
| M5 — Noise and mitigation study | Depolarising / thermal sweeps, M3 and ZNE | Out of scope for this build |
| M6 — Benchmark harness and results | Multi-seed sweeps, artifacts, figures, RESULTS.md | Planned |
| M7 — Service layer | FastAPI job submission, Docker | Out of scope for this build |
| M8 — Viewer and writeup | Streamlit artifact viewer | Out of scope for this build |

Optional dependency groups for the out-of-scope milestones (`[quantum]`, `[reference]`)
are declared in `pyproject.toml` so the boundary is explicit rather than implied, but
nothing in the package imports them yet.

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
