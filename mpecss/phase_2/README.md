# `mpecss/phase_2/` — Phase II: Homotopy Solver (Scholtes Regularization)

This subpackage implements Phase II of the MPECSS algorithm: the core homotopy loop that drives the regularization parameter `t_k → 0`.

## Modules

| Module | Description |
|---|---|
| `homotopy.py` | `run_mpecss()` — the main solver entry point; orchestrates all three phases |
| `config.py` | `DEFAULT_PARAMS` dictionary and `merge_params()` utility |
| `sign_test.py` | `evaluate_iteration_stationarity()` — checks multiplier sign conditions for S-stationarity |
| `t_update.py` | `compute_next_t()` — adaptive regularization parameter update with multiple regimes |

## How It Works

At each outer iteration:

1. **Solve** a smoothed NLP where `G(x)·H(x) ≤ t_k` replaces the hard complementarity
2. **Evaluate** the sign test to detect S-stationarity
3. **Update** `t_k` using adaptive rules based on convergence progress
4. **Track** the best iterate (lowest complementarity residual)
5. **Exit** when convergence criteria are met or budgets exhausted

## Adaptive t-Update Regimes

| Regime | Trigger | Behaviour |
|---|---|---|
| `superlinear` | Rapid convergence detected | Aggressive t reduction |
| `fast` | Normal progress | Standard geometric: `t_{k+1} = κ·t_k` |
| `slow` | Progress slowing | Conservative reduction |
| `adaptive_jump` | Stagnation detected | Re-escalate t to escape local minimum |
| `post_stagnation_fast` | Recovery after stagnation | Resume fast reduction |

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `t0` | 1.0 | Initial regularization parameter |
| `kappa` | 0.5 | Contraction factor |
| `eps_tol` | 1e-6 | Complementarity tolerance |
| `max_outer` | 3000 | Maximum outer iterations |
| `adaptive_t` | `True` | Enable adaptive updates (vs. fixed geometric) |
| `wall_timeout` | `None` | Per-solve wall-clock budget (seconds) |

See the parent [`mpecss/README.md`](../README.md) for full documentation.
