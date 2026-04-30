# `mpecss/helpers/solver/` — NLP Solver Wrappers and Caching

This subpackage provides the NLP solver interface used by all three phases of the MPECSS algorithm.

## Architecture

```
solve_with_solver_fallback()     ← Primary entry point
├─→ solver_ipopt.py              ← Try IPOPT first (via CasADi)
│   ├── solver_ipopt_config.py   ← IPOPT option defaults
│   └── solver_ipopt_helpers.py  ← IPOPT-specific utilities
├─→ solver_sqp.py                ← Fall back to SQP if IPOPT fails
│   └── solver_sqp_options.py    ← SQP option defaults
├─→ solver_cache.py              ← Cache CasADi problem templates and solver instances
│   ├── solver_cache_keys.py     ← Cache key generation
│   └── solver_cache_store.py    ← LRU-evicting cache backend
├─→ solver_acceleration.py       ← Warm-start from previous solutions
└─→ solver_metrics.py            ← Track call counts and timing
```

## Key Functions

| Function | Module | Description |
|---|---|---|
| `solve_with_solver_fallback()` | `solver_wrapper.py` | Solve a parametric NLP for a given (z, t, delta) triple. Tries IPOPT; falls back to SQP on failure |
| `is_solver_success()` | `solver_wrapper.py` | Check if a solver status string indicates success |
| `clear_solver_cache()` | `solver_cache.py` | Clear cached templates and solvers (used between problems in benchmarks) |
| `check_memory_pressure()` | `solver_cache.py` | Check if process memory is above threshold |

## Caching Strategy

To avoid re-constructing CasADi symbolic graphs on every NLP solve:

1. **Template cache**: Caches the CasADi symbolic NLP structure for each problem
2. **Solver cache**: Caches configured IPOPT/SQP solver instances
3. **Parametric cache**: Caches problem parameterization for warm-starting

The caches use LRU eviction and are cleared between benchmark problems to prevent memory accumulation.

## Solver Output

Each solver call returns a dictionary with:

| Key | Type | Description |
|---|---|---|
| `z_k` | `ndarray` | Solution vector |
| `f_val` | `float` | Objective value |
| `status` | `str` | Solver status (e.g., `Solve_Succeeded`) |
| `lam_g` | `ndarray` | Constraint multipliers |
| `kkt_res` | `float` | KKT residual norm |
| `iter_count` | `int` | Number of solver iterations |
| `cpu_time` | `float` | Solver CPU time |
| `problem_info` | `dict` | Problem structure metadata |

See the parent [`mpecss/README.md`](../../README.md) for full documentation.
