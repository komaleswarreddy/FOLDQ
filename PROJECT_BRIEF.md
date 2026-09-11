# FoldQ — Project Brief

> **How to use this file.** Put it at the repo root and commit it before writing any code.
> Then drive Claude Code milestone by milestone:
>
> ```
> Read PROJECT_BRIEF.md in full. Do not write code yet.
> Summarise back to me: the scientific goal, the repo layout, and the
> acceptance criteria for M1. Flag anything ambiguous or anything you
> think is scientifically wrong.
> ```
>
> Then, for each milestone:
>
> ```
> Implement milestone M1 from PROJECT_BRIEF.md.
> Follow the Guardrails section exactly. Write the tests first.
> Stop when every M1 acceptance criterion passes and show me the test output.
> Do not start M2.
> ```
>
> One milestone per session. Never let it run ahead — you have to be able to
> defend every line in an interview, and you cannot defend code you watched
> get generated in a 2000-line burst.

---

## 1. What this project is

A hybrid quantum–classical solver for the coarse-grained protein folding
problem on a lattice, built to answer one question honestly:

> **For peptide folding at today's hardware scale, how do variational quantum
> algorithms actually compare against classical and quantum-inspired classical
> solvers on the identical problem instance?**

The deliverable is not "I ran VQE." It is a **reproducible benchmark** with an
honest verdict, including the cases where the quantum approach loses.

### Why this framing matters

The target employer (Ceremorphic) builds *quantum-inspired silicon* — classical
hardware that borrows quantum-derived algorithms — for in-silico drug design.
The single most valuable thing this project can demonstrate is that you
understand **why** a company would choose that path over a gate-based quantum
computer. That understanding comes from the head-to-head benchmark in M5, not
from the VQE implementation.

If you only have time for part of this project, the classical and
quantum-inspired solvers plus the benchmark harness are worth more than the
quantum solver.

---

## 2. Scientific specification

### 2.1 Physical model

Coarse-grained peptide on a **tetrahedral (diamond) lattice**. Each amino acid
is a single bead at a lattice vertex. Use the tetrahedral lattice rather than a
cubic one: it has exactly 4 outgoing directions per vertex, which maps cleanly
to 2 qubits per turn with no wasted states.

For a chain of `N` beads there are `N-1` turns. Encode each turn in 2 qubits
(directions 00, 01, 10, 11). Fix the first two turns to a canonical value to
quotient out global rotational symmetry — this removes qubits for free and is
the first thing an interviewer will ask whether you did.

Support two interaction models, selectable at runtime:

- **HP model** — beads are Hydrophobic or Polar; energy `-1` per non-bonded
  H–H contact at unit lattice distance, `0` otherwise. Start here. It has
  published ground states you can check against.
- **Miyazawa–Jernigan (MJ)** — a 20×20 empirical contact potential between
  amino acid types. Ship the MJ matrix as a data file with a citation in the
  docstring. Use this for the realistic runs.

### 2.2 Hamiltonian

Build `H = H_geom + H_chir + H_int`:

- `H_geom` — penalty terms forbidding a turn immediately reversing the previous
  turn (back-tracking) and forbidding two beads occupying the same vertex.
- `H_chir` — penalty enforcing correct side-chain chirality where applicable.
  For the main-chain-only HP model this term is empty; keep the hook so the
  model can be extended, and say so in the docstring rather than pretending
  it's implemented.
- `H_int` — the contact-energy term, gated by indicator variables that are 1
  only when beads `i` and `j` are at unit lattice distance.

Penalty weights must be **derived, not guessed**: the geometric penalty must
strictly exceed the largest possible energy gain from any interaction term, or
the ground state is not a valid conformation. Compute this bound in code, assert
it, and document the derivation in the docstring. A hardcoded `lambda = 10` is
an interview failure.

Expose the Hamiltonian in three equivalent forms, with tests proving they agree:

1. `SparsePauliOp` (for Qiskit / VQE)
2. QUBO matrix (for simulated annealing)
3. Ising `(J, h)` (for the simulated bifurcation solver)

Reference implementation for cross-checking: Robert, Barkoutsos, Woerner,
Tavernelli, *"Resource-efficient quantum algorithm for protein folding"*,
npj Quantum Information 7, 38 (2021). Qiskit Nature shipped a
`protein_folding` sampling problem based on this paper; if it is still
installable in the version you pin, use it as an **oracle in tests only** —
write your own encoding from scratch and assert the two produce the same
ground-state energy. If it has been removed from the version you pin, say so in
the README rather than silently dropping the cross-check.

### 2.3 Solvers

All four solve the **identical** Hamiltonian instance. This is non-negotiable —
the whole benchmark is meaningless if the instances differ.

| Solver | Module | Notes |
|---|---|---|
| Exhaustive enumeration | `solvers/exact.py` | Ground truth. Enumerate all `4^(N-1)` turn sequences, filter invalid, take min. Feasible to roughly N=10. Also provide exact diagonalisation of the Pauli operator for N small enough. |
| Simulated annealing | `solvers/annealing.py` | Classical baseline on the QUBO. Geometric cooling schedule, single-bit and two-bit flip moves. |
| Simulated bifurcation | `solvers/bifurcation.py` | **The quantum-inspired solver.** Implement ballistic SB (Goto et al., *Sci. Adv.* 2019/2021) in NumPy — symplectic Euler integration of the Hamiltonian dynamics, vectorised over a batch of replicas. This is the Ceremorphic-relevant piece; give it real care. |
| VQE / QAOA with CVaR | `solvers/variational.py` | Qiskit. `EfficientSU2` and a QAOA ansatz, both selectable. COBYLA and SPSA optimisers. |

**CVaR aggregation is required, not optional.** Instead of minimising the
expectation value, minimise the mean of the lowest `alpha` fraction of sampled
energies (`alpha` configurable, default 0.1–0.25). Reference: Barkoutsos et al.,
*"Improving Variational Quantum Optimization using CVaR"*, Quantum 4, 256 (2020).
Implement plain expectation-value aggregation as well, and include the
comparison in the benchmark — showing *why* CVaR helps on combinatorial
objectives is a strong interview moment.

### 2.4 Noise study

Using `qiskit_aer.noise.NoiseModel`:

- Depolarising error on 1q and 2q gates, swept across a range of strengths
- Thermal relaxation (T1/T2) — offer both a synthetic sweep and, if a real
  backend's calibration data is accessible, the measured values
- Readout error

Mitigation, each toggleable so you can report the delta each one buys:

- **Measurement error mitigation** — use `mthree` (M3), or implement a
  calibration-matrix approach yourself
- **Zero-noise extrapolation** — unitary folding (`G -> G G† G`) at noise
  scale factors [1, 3, 5], then linear and Richardson extrapolation

### 2.5 Metrics

Every benchmark run records:

- Ground-state energy found, and the gap to the exact optimum from `exact.py`
- **Approximation ratio** `E_found / E_optimal`
- **Success probability** `p_s` — fraction of independent runs reaching the
  exact optimum
- **Time-to-solution** `TTS = t_run * ln(1 - 0.99) / ln(1 - p_s)` — the standard
  metric for comparing stochastic optimisers at 99% confidence. Handle `p_s = 1`
  and `p_s = 0` explicitly rather than letting it produce `inf` or `nan`.
- Qubit count, circuit depth, and 2-qubit gate count **after transpilation**
  to a realistic coupling map (not the logical circuit — reporting logical
  depth is a common and obvious overstatement)
- Wall-clock time and function-evaluation count

Scaling curves for all of the above as a function of `N`.

---

## 3. Repo layout

```
foldq/
├── PROJECT_BRIEF.md            # this file
├── README.md                   # what it is, the results table, how to reproduce
├── RESULTS.md                  # the full benchmark writeup with figures
├── pyproject.toml              # pinned deps, ruff + mypy + pytest config
├── Makefile                    # make test | bench | figures | serve | ui
├── Dockerfile
├── .github/workflows/ci.yml
├── foldq/
│   ├── lattice.py              # tetrahedral geometry, turns -> coordinates
│   ├── peptide.py              # sequences, HP classes, MJ matrix loader
│   ├── encoding.py             # turn encoding, qubit indexing, symmetry fixing
│   ├── hamiltonian.py          # H construction; Pauli / QUBO / Ising views
│   ├── solvers/
│   │   ├── base.py             # Solver protocol: .solve(H, seed) -> SolverResult
│   │   ├── exact.py
│   │   ├── annealing.py
│   │   ├── bifurcation.py
│   │   └── variational.py
│   ├── noise/
│   │   ├── models.py
│   │   └── mitigation.py       # M3 + ZNE
│   ├── benchmark/
│   │   ├── runner.py           # sweeps, seeds, artifact writing
│   │   └── metrics.py          # TTS, approximation ratio, success probability
│   ├── viz/
│   │   ├── conformation.py     # 3D folded-chain plot
│   │   └── plots.py            # convergence, scaling, noise-degradation
│   └── cli.py                  # `foldq solve`, `foldq bench`, `foldq figures`
├── api/
│   └── main.py                 # FastAPI job submission
├── ui/
│   └── app.py                  # Streamlit viewer — reads artifacts only
├── data/
│   ├── mj_matrix.csv
│   └── sequences.yaml          # benchmark instances with known ground states
├── benchmarks/
│   ├── results/                # committed JSON artifacts
│   └── figures/                # committed PNGs
└── tests/
```

All solvers implement one `Solver` protocol returning a `SolverResult`
dataclass. The benchmark runner must be able to add a new solver without
touching the runner — if adding a solver requires editing `runner.py`, the
abstraction is wrong.

---

## 4. Milestones

Each milestone ends with passing tests and a commit. Do not proceed on a red
test suite.

### M0 — Scaffold
Package skeleton, `pyproject.toml` with pinned versions, ruff + mypy + pytest
configured, Makefile, CI running tests on push, MIT licence.

*Acceptance:* `make test` passes on an empty suite; CI is green.

### M1 — Lattice and exhaustive baseline
Tetrahedral lattice geometry, turn-sequence → 3D coordinates, self-avoidance
check, HP contact counting, exhaustive enumeration.

*Acceptance:*
- Coordinate generation is tested against hand-computed positions for N=4
- Self-avoidance correctly rejects a known self-intersecting turn sequence
- Exhaustive search reproduces the published ground-state energy for at least
  two benchmark HP sequences (cite the source in the test docstring)
- Symmetry fixing is proven to reduce the search space without changing the
  optimum — test asserts both the reduced and full enumeration give the same
  minimum energy

### M2 — Hamiltonian construction
Build `H` in all three representations with derived penalty weights.

*Acceptance:*
- Penalty-weight bound is computed and asserted, not hardcoded
- For N ≤ 6: exact diagonalisation of the Pauli operator matches the M1
  exhaustive minimum, to machine precision
- QUBO and Ising forms are proven equivalent to the Pauli form by evaluating
  all three on every bitstring for N ≤ 5
- Ground state of `H` decodes back to a valid self-avoiding conformation
- If a reference implementation is installable, a test asserts agreement

### M3 — Classical and quantum-inspired solvers
Simulated annealing and simulated bifurcation.

*Acceptance:*
- Both reach the exact optimum in ≥ 95% of 50 seeded runs for N ≤ 8
- Simulated bifurcation is vectorised over replicas and demonstrably faster
  per-sample than the annealer at equal quality — show the numbers
- All randomness is seeded and runs are bit-for-bit reproducible

### M4 — Variational quantum solver
VQE and QAOA with CVaR and expectation-value aggregation.

*Acceptance:*
- Reproduces the exact ground state for a small instance on a statevector
  simulator
- CVaR vs expectation-value comparison is recorded, with convergence curves
- Transpiled depth and 2q gate count are logged for every run
- Qubit count for your benchmark instances is reported and compared against the
  scaling in the reference paper — if yours is higher, explain why in RESULTS.md

### M5 — Noise and mitigation study
Sweep noise strength; measure degradation; apply M3 and ZNE.

*Acceptance:*
- Success probability vs noise strength curve, with and without each
  mitigation technique
- The noise level at which the variational solver drops below the classical
  annealer is identified and stated explicitly
- ZNE implementation is tested against a case with a known analytic answer

### M6 — Benchmark harness and results
Multi-seed, multi-N, multi-solver sweeps writing JSON artifacts; figure
generation; RESULTS.md.

*Acceptance:*
- `make bench` regenerates every number in README.md and RESULTS.md from scratch
- Artifacts include the git SHA, package versions, and seeds
- RESULTS.md states a clear verdict, **including where the quantum solver loses**
- A `make bench-fast` target runs a reduced sweep in under 5 minutes for CI

### M7 — Service layer
FastAPI job submission (`POST /jobs` → job id, `GET /jobs/{id}` → status and
result), background execution, Dockerfile, CI builds the image.

*Acceptance:*
- Integration test submits a job and polls to completion
- `docker run` works from a clean clone
- No secrets in the image or repo

### M8 — Streamlit viewer and writeup
Read-only viewer over `benchmarks/results/`. Instance picker, 3D conformation
plot, convergence curves, solver comparison table, noise-degradation plot.
**No solving in the UI** — there must be no code path where a button press
starts a VQE run.

*Acceptance:*
- Cold start renders in under 3 seconds with no computation
- Deployed and linked from the README
- README leads with the scientific question and the results table, not with a
  feature list

---

## 5. Guardrails

These are absolute. Restate them at the start of every session.

1. **Never fabricate a number.** Every figure in README.md and RESULTS.md must
   trace to a committed artifact generated by `make bench`. If a benchmark has
   not been run, the README says "not yet measured" — it does not contain a
   plausible-looking placeholder.
2. **Tests before implementation** for anything in `foldq/` that encodes
   physics. The test states the expected physical behaviour and cites a source
   where one exists.
3. **No magic constants.** Penalty weights, cooling schedules, and CVaR alpha
   are either derived in code with the derivation in the docstring, or exposed
   as documented configuration with the default justified.
4. **Seed everything.** Every stochastic function takes an explicit seed. Two
   runs with the same seed produce identical output.
5. **Cite in docstrings.** Every non-obvious physics or algorithm choice names
   the paper it comes from.
6. **Report transpiled circuit metrics**, never logical ones.
7. **No secrets, ever.** No API keys in code, config, or git history.
8. **Stop at the milestone boundary** and show test output. Do not continue.
9. **If you are unsure whether something is scientifically correct, say so
   and stop.** A flagged uncertainty is useful. A confident wrong Hamiltonian
   costs a week.

---

## 6. Scope and honesty

This is **4–6 weeks of real part-time work**, not a weekend. A realistic
reduced scope, if time is short:

- **Minimum defensible version:** M0–M3 plus M6. You get the lattice model,
  a validated Hamiltonian, the classical and quantum-inspired solvers, and a
  reproducible benchmark. You can then honestly describe the quantum solver as
  planned work. This is still a strong project.
- **Do not** ship M4 half-built. A VQE implementation you cannot explain is
  worse than no VQE implementation, because it invites exactly the questions
  you cannot answer.

Update your resume bullets to match what actually exists. If only M0–M3 and M6
ship, the bullets should describe the classical and quantum-inspired benchmark
and drop the VQE/QAOA and mitigation claims.

---

## 7. Interview defence checklist

You must be able to answer all of these without notes. If you cannot, you do
not understand a part of your own project and should go back to it.

**Encoding**
- Why a tetrahedral lattice rather than cubic?
- Why 2 qubits per turn, and how many qubits total for a 10-residue peptide?
- Which symmetry did you quotient out by fixing the first turns, and how many
  qubits did that save?
- What does a bitstring physically mean, and how do you decode it?

**Hamiltonian**
- How did you derive the penalty weight, and what breaks if it is too small?
- Why does the interaction term need indicator variables?
- How do you know your Hamiltonian is correct?

**Algorithms**
- What is CVaR aggregation and why does it beat the expectation value for a
  combinatorial objective?
- What is the difference between VQE and QAOA here?
- Explain simulated bifurcation. Why is it called quantum-inspired if it runs
  entirely on classical hardware?
- What is a barren plateau and did you hit one?

**Results**
- At what noise level does your quantum solver lose to simulated annealing?
- What does zero-noise extrapolation assume, and when does that assumption fail?
- Why is time-to-solution the right comparison metric rather than wall-clock?
- What is the honest verdict of your benchmark?

**The one that matters most**
- *"Given your own results, why would a company build quantum-inspired
  classical silicon instead of a quantum computer?"*

Your answer to that last question is the interview. Everything else in this
project exists to let you answer it from data you generated yourself.