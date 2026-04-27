from typing import Any, Dict

_DEFAULT_MAX_OUTER = 3000

DEFAULT_PARAMS: Dict[str, Any] = {
    "t0": 1.0,
    "kappa": 0.5,
    "eps_tol": 1e-6,  # Standard MPEC tolerance (1e-6 common in papers)
    "acceptable_tol": 1e-5,  # Accept solutions with comp_res below this (per IPOPT convention)
    "delta_k": 0.0,
    "max_outer": _DEFAULT_MAX_OUTER,

    "tau": 1e-6,
    "sta_tol": None,
    "adaptive_t": True,
    "stagnation_window": 10,
    "solver_opts": None,
    "log_csv": None,
    "seed": 0,
    "feasibility_phase": True,
    "phase1_max_attempts": 3,
    "phase1_random_restarts": 3,
    "restoration_strategy": "cascade",
    "restoration_enabled": True,
    "perturb_eps": 0.01,
    "gamma": 1.0,
    "step_size": 0.1,
    "max_restorations": 50,        # hard cap on total restoration calls per solve
    "restoration_stag_window": 8,  # consecutive restorations with <0.01% comp_res
    "wall_timeout": None,          # per-solve wall-clock budget in seconds (None = unlimited)
    "max_adaptive_jumps": 500,     # hard cap on adaptive_jump regime triggers per solve
    "restoration_comp_factor": 10,  # Only restore if comp_res > factor * eps_tol
    "high_restoration_skip_threshold": 10,  # Skip final push if n_restorations >= this
    "early_c_phase2_enabled": True,
    "early_c_phase2_iters_small": 12,
    "early_c_phase2_iters_large": 20,
    "early_probe_phase2_iters_small": 20,
    "early_probe_phase2_iters_medium": 24,
    "early_probe_phase2_iters_large": 12,
}

def merge_params(params: Dict[str, Any] | None) -> Dict[str, Any]:
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    return p
