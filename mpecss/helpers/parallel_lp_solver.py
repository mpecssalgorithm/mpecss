# Parallel LP Solver for Phase III B-Stationarity Certification.

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

logger = logging.getLogger('mpecss.parallel_lp')

_PARALLEL_THRESHOLD_BRANCHES = 64  # 2^6 = 64 branches
_MAX_WORKERS = None  # None = use os.cpu_count()


def _get_num_workers() -> int:
    # Get optimal number of worker threads for LP solving.
    import os
    cpu_count = os.cpu_count() or 4
    return min(cpu_count, 32)


def solve_single_lp_branch(
    branch_idx: int,
    grad_f: np.ndarray,
    A_ub_base: List[np.ndarray],
    b_ub_base: List[float],
    bounds: List[Tuple[float, float]],
    I_B: List[int],
    J_G: np.ndarray,
    J_H: np.ndarray,
) -> Tuple[int, float, Optional[np.ndarray], bool]:
    # Solve a single branch LP.
    from scipy.optimize import linprog

    A_ub_branch = list(A_ub_base)
    b_ub_branch = list(b_ub_base)

    for bit_pos, i in enumerate(I_B):
        if (branch_idx >> bit_pos) & 1:
            A_ub_branch.append(-J_G[i])
            b_ub_branch.append(0)
        else:
            A_ub_branch.append(-J_H[i])
            b_ub_branch.append(0)

    if len(A_ub_branch) > 0:
        A_ub = np.vstack(A_ub_branch)
    else:
        A_ub = None

    try:
        result = linprog(
            grad_f,
            A_ub=A_ub,
            b_ub=b_ub_branch if b_ub_branch else None,
            bounds=bounds,
            method='highs'
        )
        if result.success:
            return (branch_idx, result.fun, result.x.copy(), True)
        else:
            return (branch_idx, float('inf'), None, False)
    except Exception as e:
        logger.debug(f'LP solve failed for branch {branch_idx}: {e}')
        return (branch_idx, float('inf'), None, False)


def solve_bstationarity_parallel(
    biactive_branches: range,
    grad_f: np.ndarray,
    A_ub_base: List[np.ndarray],
    b_ub_base: List[float],
    bounds: List[Tuple[float, float]],
    I_B: List[int],
    J_G: np.ndarray,
    J_H: np.ndarray,
    eps_bstat: float = 1e-8,
    timeout: float = 60.0,
) -> Tuple[bool, float, Optional[np.ndarray], int, Dict[str, Any]]:
    # Solve B-stationarity LP enumeration in parallel.
    n_branches = len(biactive_branches)
    t_start = time.time()

    if n_branches < _PARALLEL_THRESHOLD_BRANCHES:
        return _solve_sequential(
            biactive_branches, grad_f, A_ub_base, b_ub_base, bounds,
            I_B, J_G, J_H, eps_bstat, timeout
        )

    n_workers = _get_num_workers()
    logger.info(f'Phase III: Parallel LP solving with {n_workers} workers for {n_branches} branches')

    best_obj = 0.0
    best_direction = None
    best_branch = -1
    completed = 0
    timed_out = False
    early_exit = False

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                solve_single_lp_branch,
                branch_idx, grad_f, A_ub_base, b_ub_base, bounds, I_B, J_G, J_H
            ): branch_idx
            for branch_idx in biactive_branches
        }

        for future in as_completed(futures):
            if time.time() - t_start > timeout:
                timed_out = True
                for f in futures:
                    f.cancel()
                break

            branch_idx, obj_val, direction, success = future.result()
            completed += 1

            if success and obj_val < best_obj:
                best_obj = obj_val
                best_direction = direction
                best_branch = branch_idx

                if best_obj < -eps_bstat * 100:
                    early_exit = True
                    logger.debug(f'Early exit: found descent direction with obj={best_obj:.2e}')
                    for f in futures:
                        f.cancel()
                    break

    is_bstat = best_obj >= -eps_bstat

    details = {
        'lpec_status': 'timed_out' if timed_out else ('early_exit' if early_exit else 'complete'),
        'branches_enumerated': completed,
        'total_branches': n_branches,
        'n_workers': n_workers,
        'wall_time': time.time() - t_start,
        'parallel': True,
    }

    return is_bstat, best_obj, best_direction, best_branch, details


def _solve_sequential(
    biactive_branches: range,
    grad_f: np.ndarray,
    A_ub_base: List[np.ndarray],
    b_ub_base: List[float],
    bounds: List[Tuple[float, float]],
    I_B: List[int],
    J_G: np.ndarray,
    J_H: np.ndarray,
    eps_bstat: float,
    timeout: float,
) -> Tuple[bool, float, Optional[np.ndarray], int, Dict[str, Any]]:
    # Sequential fallback for small branch counts.
    t_start = time.time()
    best_obj = 0.0
    best_direction = None
    best_branch = -1
    completed = 0
    timed_out = False

    for branch_idx in biactive_branches:
        if time.time() - t_start > timeout:
            timed_out = True
            break

        branch_idx, obj_val, direction, success = solve_single_lp_branch(
            branch_idx, grad_f, A_ub_base, b_ub_base, bounds, I_B, J_G, J_H
        )
        completed += 1

        if success and obj_val < best_obj:
            best_obj = obj_val
            best_direction = direction
            best_branch = branch_idx

    is_bstat = best_obj >= -eps_bstat

    details = {
        'lpec_status': 'timed_out' if timed_out else 'complete',
        'branches_enumerated': completed,
        'total_branches': len(biactive_branches),
        'n_workers': 1,
        'wall_time': time.time() - t_start,
        'parallel': False,
    }

    return is_bstat, best_obj, best_direction, best_branch, details
