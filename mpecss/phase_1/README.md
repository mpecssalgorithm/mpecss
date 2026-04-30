# `mpecss/phase_1/` — Phase I: Feasibility Search

This subpackage implements Phase I of the MPECSS algorithm: finding a feasible starting point.

## Modules

| Module | Description |
|---|---|
| `feasibility.py` | `run_feasibility_phase()` — orchestrates multi-start feasibility search with random restarts |
| `feasibility_nlp.py` | Constructs the CasADi NLP for the feasibility problem (minimize complementarity residual subject to original constraints) |

## How It Works

1. Solve a minimum-complementarity NLP from the user-supplied initial point `z0`
2. If the first attempt does not achieve sufficient feasibility, perform random restarts (controlled by `phase1_random_restarts`)
3. Return the best feasible point found, along with diagnostic metadata

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `feasibility_phase` | `True` | Enable/disable Phase I |
| `phase1_max_attempts` | 3 | Maximum number of NLP solve attempts |
| `phase1_random_restarts` | 3 | Number of random restart points to try |
| `seed` | 0 | Random seed for restart generation |

## Returns

A dictionary with keys: `z_feasible`, `success`, `cpu_time`, `ipopt_iter_count`, `initial_comp_res`, `final_comp_res`, `feasibility_achieved`, etc.

See the parent [`mpecss/README.md`](../README.md) for full documentation.
