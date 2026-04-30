# `mpecss/` — Core Solver Package

This directory contains the complete MPECSS solver implementation. The solver uses a three-phase Scholtes-type regularization approach to solve Mathematical Programs with Equilibrium Constraints (MPECs / MPCCs).

---

## Package Entry Point

- **`__init__.py`** — Exposes `run_mpecss()` as the top-level API. Import with:

  ```python
  from mpecss import run_mpecss
  ```

- **`contracts.py`** — Typed contracts used across the solver:
  - `SolverStatus` (enum): `CONVERGED`, `MAX_ITER`, `NLP_FAILURE`, `TIMEOUT`, `STAGNATION`, `STATIONARITY_UNVERIFIABLE`, `UNSUPPORTED_MODEL`, `OOM`, `CRASHED`, `LOAD_FAILED`
  - `StationarityClass` (enum): `B_STATIONARY`, `C_STATIONARY`, `FAIL`
  - `ProblemSpec` (TypedDict): The expected shape of an MPEC problem dictionary returned by any loader (see `helpers/loaders/`).
  - `SolveResult` (TypedDict): The expected shape of the result dictionary returned by `run_mpecss()`. See [`contracts.py`](contracts.py) for the full list of fields.

---

## Subpackage Layout

```
mpecss/
├── __init__.py           # Package entry point; re-exports run_mpecss()
├── contracts.py          # SolverStatus, StationarityClass, ProblemSpec, SolveResult
│
├── phase_1/              # Phase I: Feasibility
│   ├── __init__.py
│   ├── feasibility.py    # Orchestrates multi-start feasibility search
│   └── feasibility_nlp.py # NLP formulation for the Phase I feasibility problem
│
├── phase_2/              # Phase II: Homotopy (Scholtes regularization)
│   ├── __init__.py
│   ├── config.py         # DEFAULT_PARAMS dict and merge_params() utility
│   ├── homotopy.py       # Main solver loop: run_mpecss()
│   ├── sign_test.py      # Sign-test stationarity check (S-stationarity detection)
│   └── t_update.py       # Adaptive regularization parameter (t_k) update rules
│
├── phase_3/              # Phase III: Solution polishing and B-stationarity certification
│   ├── __init__.py
│   ├── bnlp_polish.py    # BNLP (Branch NLP) polishing with active-set identification
│   ├── bnlp_polish_sets.py # Active-set identification helpers for BNLP
│   ├── bnlp_polish_utils.py # Utility functions for BNLP polishing
│   ├── bstationarity.py  # B-stationarity certification via LPEC enumeration + LICQ check
│   └── lpec_refine.py    # LPEC-guided refinement loop (optional post-processing)
│
├── benchmark/            # Benchmark orchestration utilities
│   ├── __init__.py
│   ├── benchmark_utils.py    # Main benchmark runner: run_benchmark_main(), run_single_problem_internal()
│   ├── benchmark_results.py  # Result post-processing, diagnostic columns, snapshot mapping
│   ├── benchmark_failure.py  # Failure-mode handling and problem-size classification
│   └── benchmark_audit.py    # Audit trail recorder for provenance tracking
│
└── helpers/              # Shared utilities (loaders, solver wrappers, monitoring)
    ├── __init__.py
    ├── comp_residuals.py     # Complementarity residual computation: min(G(x), H(x))
    ├── utils.py              # IterationLog dataclass, export_csv() helper
    ├── monitoring_system.py  # System resource monitoring (CPU, memory)
    ├── monitoring_timeout.py # Timeout enforcement and wall-clock budget tracking
    ├── preflight_checks.py   # Pre-run environment validation (Python version, deps, paths)
    │
    ├── loaders/              # Problem instance loaders
    │   ├── __init__.py       # Re-exports all three loader functions
    │   ├── mpeclib_loader.py # load_mpeclib(path) → ProblemSpec
    │   ├── macmpec_loader.py # load_macmpec(path) → ProblemSpec
    │   └── nosbench_loader.py# load_nosbench(path) → ProblemSpec
    │
    └── solver/               # NLP solver wrappers and caching
        ├── __init__.py
        ├── solver_wrapper.py     # solve_with_solver_fallback(): tries IPOPT, falls back to SQP
        ├── solver_ipopt.py       # IPOPT solver interface via CasADi
        ├── solver_ipopt_config.py# IPOPT option configuration
        ├── solver_ipopt_helpers.py# IPOPT-specific helper functions
        ├── solver_sqp.py         # SQP solver interface (fallback solver)
        ├── solver_sqp_options.py # SQP option configuration
        ├── solver_cache.py       # Solver-level caching (template, parametric, solution)
        ├── solver_cache_keys.py  # Cache key generation
        ├── solver_cache_store.py # Cache storage backend
        ├── solver_acceleration.py# Warm-start and acceleration utilities
        └── solver_metrics.py     # Solver performance metrics collection
```

---

## The Three-Phase Algorithm

### Phase I: Feasibility (`phase_1/`)

**Goal**: Find a starting point that approximately satisfies the MPEC constraints (general constraints + complementarity).

- **`feasibility.py`** — `run_feasibility_phase()` runs a multi-start NLP feasibility search:
  1. Solves a minimum-complementarity NLP from the user-supplied `z0`.
  2. Optionally performs random restarts (controlled by `phase1_random_restarts` parameter) to escape poor initial points.
  3. Returns the best feasible point found along with diagnostic metadata (initial/final complementarity residual, CPU time, number of attempts).

- **`feasibility_nlp.py`** — Constructs the CasADi NLP for the feasibility problem. The objective is to minimize complementarity residual subject to the original constraints and variable bounds.

### Phase II: Homotopy (`phase_2/`)

**Goal**: Drive the regularization parameter `t_k → 0` using Scholtes-type smoothing while tracking the NLP solution path toward a stationary point.

- **`homotopy.py`** — `run_mpecss()` is the main solver function. It:
  1. Calls Phase I (if enabled) to obtain a feasible starting point.
  2. Performs an early S-stationarity check — if the Phase I solution is already near-optimal, Phase II can be shortened or skipped.
  3. Iteratively solves parametric NLPs with decreasing `t_k` values, using product-form Scholtes smoothing: `G(x) · H(x) ≤ t`.
  4. At each iteration, evaluates the sign test to detect S-stationarity.
  5. Tracks the best iterate (lowest complementarity residual) and handles stagnation, adaptive jumps, and wall-clock budgets.
  6. Passes the best solution to Phase III for certification.

- **`config.py`** — `DEFAULT_PARAMS` dictionary defines all tunable parameters:

  | Parameter | Default | Description |
  |---|---|---|
  | `t0` | 1.0 | Initial regularization parameter |
  | `kappa` | 0.5 | Contraction factor for t_k reduction |
  | `eps_tol` | 1e-6 | Complementarity tolerance for convergence |
  | `max_outer` | 3000 | Maximum Phase II outer iterations |
  | `tau` | 1e-6 | Stationarity tolerance scaling factor |
  | `adaptive_t` | True | Enable adaptive t_k update (vs. fixed geometric reduction) |
  | `stagnation_window` | 10 | Window size for stagnation detection |
  | `feasibility_phase` | True | Enable/disable Phase I |
  | `wall_timeout` | None | Per-solve wall-clock budget in seconds |
  | `seed` | 0 | Random seed for Phase I restarts |

- **`sign_test.py`** — `evaluate_iteration_stationarity()` checks the multiplier signs at the current iterate. If all complementarity multipliers satisfy the sign conditions, the point is S-stationary (a necessary condition for B-stationarity).

- **`t_update.py`** — `compute_next_t()` implements adaptive regularization parameter updates with several regimes:
  - **superlinear**: Fast reduction when rapid convergence is detected
  - **fast**: Standard geometric reduction (`t_k+1 = kappa * t_k`)
  - **slow**: Conservative reduction when progress slows
  - **adaptive_jump**: Re-escalation when stagnation is detected
  - **post_stagnation_fast**: Recovery after stagnation episodes

### Phase III: Polishing & Certification (`phase_3/`)

**Goal**: Polish the converged solution and certify whether it achieves B-stationarity.

- **`bnlp_polish.py`** — `bnlp_polish()` performs Branch-NLP polishing:
  1. Identifies the active set at the solution (which complementarity pairs are active on G vs. H side).
  2. Solves a "branch NLP" with the active complementarity constraints fixed, yielding a cleaner solution.
  3. Tries alternative partitions of biactive indices to find the best branch.

- **`bnlp_polish_sets.py`** — `identify_active_set()` partitions complementarity indices into:
  - `I1`: G-active (G ≈ 0, H > 0)
  - `I2`: H-active (H ≈ 0, G > 0)
  - `I_biactive`: Both G ≈ 0 and H ≈ 0
  - `I3`: Neither active (residual pair)

- **`bstationarity.py`** — `certify_bstationarity()` performs B-stationarity certification:
  1. **LICQ check** (`check_mpec_licq()`): Verifies Linear Independence Constraint Qualification at the solution. Under LICQ, S-stationarity ⟺ B-stationarity, providing a shortcut.
  2. **LPEC enumeration**: When LICQ does not hold, solves Linear Programs with Equilibrium Constraints (LPECs) over all possible biactive partitions to verify no descent direction exists.
  3. Returns classification: B-stationary, C-stationary, or FAIL.

- **`lpec_refine.py`** — `lpec_refinement_loop()` is an optional post-processing step that iteratively refines the solution using LPEC-guided branching.

---

## Benchmark Orchestration (`benchmark/`)

The `benchmark/` subpackage provides infrastructure for running large-scale benchmark campaigns:

- **`benchmark_utils.py`** — `run_benchmark_main()` is the CLI entry point for batch runs. It:
  - Parses command-line arguments (tag, seed, workers, timeout, etc.)
  - Discovers JSON problem files in the benchmark directory
  - Supports resume from previous CSV, retry of failed problems, and problem-list filtering
  - Dispatches problems to parallel workers using `multiprocessing`
  - Writes results to a wide-format CSV with 200+ columns capturing every aspect of the solve

- **`benchmark_results.py`** — Internal post-processing utilities for result rows (all functions are module-private):
  - `_summarize_result_state()`: Extracts key metrics from a SolveResult
  - `_apply_raw_summary_columns()`: Populates `raw_*` diagnostic columns
  - `_apply_point_diagnostic_columns()`: Computes independent point-quality diagnostics (constraint violations, variable bound violations, etc.)
  - `_preserve_stronger_raw_certificate()`: Ensures that if the raw run achieved a stronger stationarity certificate than post-processing, the stronger result is kept

- **`benchmark_failure.py`** — Handles failure modes (OOM, crash, timeout, load failures) and builds standardized failure result rows.

- **`benchmark_audit.py`** — `_BenchmarkAuditRecorder` (internal) writes a JSON audit trail for each problem, recording every stage transition, timing, and intermediate results. This ensures full provenance tracking.

---

## Helpers (`helpers/`)

### Loaders (`helpers/loaders/`)

Each loader reads a JSON problem file from the respective benchmark suite and returns a `ProblemSpec` dictionary:

- **`mpeclib_loader.py`** — `load_mpeclib(path)`: Loads MPECLib `.nl.json` files converted from GAMS source.
- **`macmpec_loader.py`** — `load_macmpec(path)`: Loads MacMPEC `.nl.json` files converted from AMPL source.
- **`nosbench_loader.py`** — `load_nosbench(path)`: Loads NOSBENCH `.json` files from the CasADi format.

All loaders return a dictionary with keys: `name`, `n_x`, `n_comp`, `n_con`, `n_p`, `family`, `x0_fn`, `f_fn`, `G_fn`, `H_fn`, `build_casadi`, `lbx`, `ubx`, `lbg`, `ubg`. Some problems may include `unsupported_model_reason` if the model structure is not fully supported.

### Solver Wrappers (`helpers/solver/`)

- **`solver_wrapper.py`** — `solve_with_solver_fallback()`: Primary interface. Tries IPOPT first; if IPOPT fails, falls back to the SQP solver.
- **`solver_ipopt.py`** — CasADi-based IPOPT interface for solving parametric NLPs.
- **`solver_sqp.py`** — Custom SQP solver used as a fallback.
- **`solver_cache.py`** — Caches CasADi problem templates and solver instances to avoid expensive re-construction. Includes memory-pressure-aware eviction.
- **`solver_metrics.py`** — Tracks solver call counts, iteration totals, and timing.

### Other Helpers

- **`comp_residuals.py`** — `complementarity_residual(z, problem)`: Computes `max_i min(G_i(z), H_i(z))` — the infinity-norm complementarity residual.
- **`utils.py`** — `IterationLog` dataclass for recording per-iteration diagnostics; `export_csv()` for writing iteration logs.
- **`preflight_checks.py`** — Verifies Python version (≥3.10), required packages (`numpy`, `pandas`, `scipy`, `casadi`, `psutil`), and repository layout before a benchmark run.
- **`monitoring_system.py`** / **`monitoring_timeout.py`** — System resource monitoring and timeout enforcement.

---

## How to Use the Solver Programmatically

```python
from mpecss import run_mpecss
from mpecss.helpers.loaders import load_mpeclib  # or load_macmpec, load_nosbench

# 1. Load a problem
problem = load_mpeclib("benchmarks/mpeclib/mpeclib-json/bard1.nl.json")

# 2. Generate an initial point
z0 = problem["x0_fn"](seed=42)

# 3. Solve
result = run_mpecss(problem, z0, params={
    "t0": 1.0,
    "kappa": 0.5,
    "eps_tol": 1e-6,
    "max_outer": 3000,
    "feasibility_phase": True,
    "seed": 42,
})

# 4. Inspect result
print(f"Status:       {result['status']}")
print(f"Stationarity: {result['stationarity']}")
print(f"Comp. res.:   {result['comp_res']:.3e}")
print(f"Objective:    {result['f_final']:.6f}")
print(f"B-stationary: {result['b_stationarity']}")
print(f"LICQ holds:   {result['licq_holds']}")
```

For command-line usage (batch benchmarks, resume, retry), see the [`kaggle_setup/README.md`](../kaggle_setup/README.md) or the root [`README.md`](../README.md#running-benchmarks-locally-alternative-to-kaggle).

---

## Dependencies

The solver requires the packages listed below. For exact version pins used in paper reproduction, see [`requirements-lock.txt`](../requirements-lock.txt) and [`pyproject.toml`](../pyproject.toml) at the repository root.

| Package | Version | Purpose |
|---|---|---|
| `casadi` | ≥ 3.6.3 | Automatic differentiation and NLP solver interface (IPOPT) |
| `numpy` | ≥ 1.24 | Numerical arrays |
| `pandas` | ≥ 2.0 | CSV I/O for benchmark results |
| `scipy` | ≥ 1.11 | Sparse matrix operations (Jacobian evaluation) |
| `psutil` | ≥ 5.9 | System memory monitoring |
| `pytest` | ≥ 7.4 | Testing (optional, install via `pip install -e ".[test]"`) |
