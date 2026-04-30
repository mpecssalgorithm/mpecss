# `mpecss/helpers/` — Shared Utilities

This subpackage provides shared utilities used across all three solver phases and the benchmark infrastructure.

## Modules

| Module | Description |
|---|---|
| `comp_residuals.py` | `complementarity_residual(z, problem)` — computes `max_i min(G_i(z), H_i(z))` |
| `utils.py` | `IterationLog` dataclass for per-iteration diagnostics; `export_csv()` for writing iteration logs |
| `preflight_checks.py` | Pre-run validation: Python version ≥ 3.10, required packages, repository layout, Kaggle working directory permissions |
| `monitoring_system.py` | System resource monitoring (CPU usage, memory consumption) |
| `monitoring_timeout.py` | Timeout enforcement and wall-clock budget tracking |

## Sub-packages

### `loaders/` — Problem Instance Loaders

Each loader reads a JSON problem file and returns a `ProblemSpec` dictionary:

| Module | Function | Format |
|---|---|---|
| `mpeclib_loader.py` | `load_mpeclib(path)` | `.nl.json` (GAMS-derived) |
| `macmpec_loader.py` | `load_macmpec(path)` | `.nl.json` (AMPL-derived) |
| `nosbench_loader.py` | `load_nosbench(path)` | `.json` (CasADi native) |

All loaders return a dictionary with keys: `name`, `n_x`, `n_comp`, `n_con`, `n_p`, `family`, `x0_fn`, `f_fn`, `G_fn`, `H_fn`, `build_casadi`, `lbx`, `ubx`, `lbg`, `ubg`.

### `solver/` — NLP Solver Wrappers and Caching

| Module | Description |
|---|---|
| `solver_wrapper.py` | `solve_with_solver_fallback()` — primary interface; tries IPOPT, falls back to SQP |
| `solver_ipopt.py` | CasADi-based IPOPT interface |
| `solver_ipopt_config.py` | IPOPT option configuration |
| `solver_ipopt_helpers.py` | IPOPT-specific helper functions |
| `solver_sqp.py` | Custom SQP solver (fallback) |
| `solver_sqp_options.py` | SQP option configuration |
| `solver_cache.py` | Problem template and solver instance caching with memory-aware eviction |
| `solver_cache_keys.py` | Cache key generation |
| `solver_cache_store.py` | Cache storage backend |
| `solver_acceleration.py` | Warm-start and acceleration utilities |
| `solver_metrics.py` | Solver performance metrics collection |

See the parent [`mpecss/README.md`](../README.md) for full documentation.
