# The Toolbox: Helpful tools for logging and math.

import logging
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import numpy as np
import casadi as ca




@dataclass
class IterationLog:
    # The Flight Recorder.
    iteration: int = 0
    t_k: float = 0.0
    delta_k: float = 0.0
    comp_res: float = float('inf')
    kkt_res: float = float('inf')
    objective: float = float('inf')
    sign_test: str = 'N/A'
    sign_test_reason: str = ''
    restoration_used: str = 'none'
    solver_status: str = ''
    cpu_time: float = 0.0
    n_biactive: int = 0
    t_update_regime: str = ''
    nlp_iter_count: int = 0
    solver_type: str = ''
    warmstart_type: str = 'none'
    sta_tol: float = 0.0
    improvement_ratio: float = 0.0
    restoration_trigger_reason: str = 'none'
    restoration_success: bool = False
    biactive_indices_str: str = ''
    stagnation_count: int = 0
    tracking_count: int = 0
    is_in_tracking_regime: bool = False
    solver_fallback_occurred: bool = False
    consecutive_solver_failures: int = 0
    best_comp_res_so_far: float = float('inf')
    best_iter_achieved: int = -1
    ipopt_tol_used: float = 1e-06
    lambda_G_min: float = 0.0
    lambda_G_max: float = 0.0
    lambda_H_min: float = 0.0
    lambda_H_max: float = 0.0
    z_k: Optional[np.ndarray] = field(default=None, repr=False)
    lambda_G: Optional[np.ndarray] = field(default=None, repr=False)
    lambda_H: Optional[np.ndarray] = field(default=None, repr=False)

    def to_row(self) -> dict:
        # Return a CSV-exportable dict (large array fields excluded).
        d = asdict(self)
        for key in ('z_k', 'lambda_G', 'lambda_H'):
            d.pop(key, None)
        return d


def extract_multipliers(lam_g, n_comp, problem_info):
    # Harvesting the Forces (Multipliers).
    lam_g = np.asarray(lam_g).flatten()
    lam_len = len(lam_g)  # Bounds check: total length of multiplier vector

    n_orig_con  = problem_info.get('n_orig_con', 0)
    n_bounded_G = problem_info.get('n_bounded_G', n_comp)   # default: NCP

    if 'off_G_lb' in problem_info and 'off_H_lb' in problem_info:
        off_G_lb = problem_info['off_G_lb']
        off_H_lb = problem_info['off_H_lb']
    else:
        off_G_lb = n_orig_con
        off_H_lb = off_G_lb + n_comp

    if n_bounded_G > 0:
        end_G = min(off_G_lb + n_bounded_G, lam_len)
        if off_G_lb < lam_len:
            lambda_G = -lam_g[off_G_lb : end_G]
        else:
            lambda_G = np.zeros(0)

        if len(lambda_G) < n_comp:
            full_lG = np.zeros(n_comp)
            bounded_idx = problem_info.get('_bounded_G_idx', list(range(n_bounded_G)))
            for k, i in enumerate(bounded_idx):
                if k < len(lambda_G):
                    full_lG[i] = lambda_G[k]
            lambda_G = full_lG
    else:
        lambda_G = np.zeros(n_comp)

    end_H = min(off_H_lb + n_comp, lam_len)
    if off_H_lb < lam_len:
        lambda_H = -lam_g[off_H_lb : end_H]
    else:
        lambda_H = np.zeros(n_comp)
    if len(lambda_H) < n_comp:
        lambda_H = np.pad(lambda_H, (0, n_comp - len(lambda_H)), mode='constant')

    return lambda_G, lambda_H




def multiplier_sign_test(lambda_G, lambda_H, biactive_idx, tau=1e-6):
    # S-stationarity sign check at biactive indices.
    if len(biactive_idx) == 0:
        return (True, 'no_biactive')

    reasons = []
    for i in biactive_idx:
        if lambda_G[i] < -tau:
            reasons.append(f'lam_G[{i}]={lambda_G[i]:.2e}<-{tau:.2e}')
        if lambda_H[i] < -tau:
            reasons.append(f'lam_H[{i}]={lambda_H[i]:.2e}<-{tau:.2e}')

    if not reasons:
        return (True, 'PASS')

    return (False, 'FAIL: ' + '; '.join(reasons))


def export_csv(logs: List[IterationLog], filepath: str):
    # Export iteration logs to CSV. Creates the output directory if needed.
    import pandas as pd
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    if logs:
        df = pd.DataFrame([log.to_row() for log in logs])
    else:
        df = pd.DataFrame(columns=list(IterationLog().to_row().keys()))
    df.to_csv(filepath, index=False)


