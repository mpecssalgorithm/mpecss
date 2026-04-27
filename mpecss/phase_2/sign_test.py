# Stationarity Testing: Measuring the "Quality" of our Answer.

from typing import Any, Dict, Tuple, cast
import numpy as np
from mpecss.helpers.comp_residuals import biactive_indices, biactive_residual
from mpecss.helpers.utils import extract_multipliers, multiplier_sign_test


def evaluate_iteration_stationarity(z_k, lam_g, problem, problem_info, n_comp, t_k, sta_tol, tau, biactive_tol_floor=1e-8):
    # Step-by-Step Quality Check:
    if sta_tol is None:
        sta_tol = max(biactive_tol_floor, np.sqrt(t_k))

    lambda_G, lambda_H = cast(
        Tuple[np.ndarray, np.ndarray],
        extract_multipliers(lam_g, n_comp, problem_info)
    )

    biactive_idx = biactive_indices(z_k, problem, sta_tol)

    comp_res = biactive_residual(z_k, problem)

    sign_pass, sign_reason = cast(
        Tuple[bool, str],
        multiplier_sign_test(lambda_G, lambda_H, biactive_idx, tau=tau)
    )

    return {
        'lambda_G': lambda_G,
        'lambda_H': lambda_H,
        'sta_tol': sta_tol,
        'biactive_idx': biactive_idx,
        'n_biactive': len(biactive_idx),
        'comp_res': comp_res,
        'sign_pass': sign_pass,
        'sign_reason': sign_reason,
    }
