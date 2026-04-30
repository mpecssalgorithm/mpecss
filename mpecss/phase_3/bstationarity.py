"""B-stationarity certification utilities for MPEC solutions."""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from collections import OrderedDict
import numpy as np
import casadi as ca
from mpecss.contracts import ProblemSpec
from mpecss.helpers.comp_residuals import complementarity_residual, mcp_feasibility_residual

logger = logging.getLogger('mpecss.bstationarity')

_ACTIVE_TOL = 1e-06
_BSTAT_TOL = 1e-08
_LICQ_TOL = 1e-08
_DIR_BOUND = 1.0
_BSTAT_TIMEOUT = 60.0

MAX_BSTAT_JACOBIAN_CACHE_SIZE = 30  # Keep last 30 problem Jacobians


class _BstatJacobianLRUCache:
    # Simple LRU cache for B-stationarity Jacobian functions.

    def __init__(self, max_size: int):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._evictions = 0

    def get(self, key: str):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value):
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
            return

        while len(self._cache) >= self._max_size:
            evicted_key, evicted_val = self._cache.popitem(last=False)
            self._evictions += 1
            logger.debug(f"B-stat Jacobian cache eviction: removed '{evicted_key}'")
            del evicted_val

        self._cache[key] = value

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)


_JACOBIAN_CACHE = _BstatJacobianLRUCache(MAX_BSTAT_JACOBIAN_CACHE_SIZE)

def clear_jacobian_cache():
    # Clear the Jacobian cache to free memory.
    _JACOBIAN_CACHE.clear()


def _unsupported_certificate_reason(problem: ProblemSpec) -> Optional[str]:
    return None


def _compute_jacobians(z: np.ndarray, problem: ProblemSpec):
    # Compute Jacobians of f, g_orig, G, H at point z.
    n_x = problem['n_x']
    z = np.asarray(z).flatten()
    prob_name = problem.get('name', 'unknown')

    n_comp = problem.get('n_comp', 0)
    n_con = problem.get('n_con', 0)
    family = problem.get('family', 'unknown')
    cache_key = f"{prob_name}|{family}|{n_x}|{n_comp}|{n_con}"

    cached = _JACOBIAN_CACHE.get(cache_key)
    if cached is None:
        _sym = ca.MX.sym if n_x >= 500 else ca.SX.sym
        x_sym = _sym('x', n_x)

        info = problem['build_casadi'](0, 0)
        grad_f_expr = ca.jacobian(info['f'], info['x'])
        grad_f_fn = ca.Function('grad_f', [info['x']], [grad_f_expr])

        G_expr = problem['G_fn'](x_sym)
        jac_G_fn = ca.Function('jac_G', [x_sym], [ca.jacobian(G_expr, x_sym)])

        H_expr = problem['H_fn'](x_sym)
        jac_H_fn = ca.Function('jac_H', [x_sym], [ca.jacobian(H_expr, x_sym)])

        jac_g_fn = None
        if n_con > 0:
            g_orig_expr = info['g'][:n_con]
            jac_g_fn = ca.Function('jac_g', [info['x']], [ca.jacobian(g_orig_expr, info['x'])])

        _JACOBIAN_CACHE.put(cache_key, (grad_f_fn, jac_G_fn, jac_H_fn, jac_g_fn))
        cached = (grad_f_fn, jac_G_fn, jac_H_fn, jac_g_fn)

    grad_f_fn, jac_G_fn, jac_H_fn, jac_g_fn = cached

    grad_f = np.asarray(grad_f_fn(z)).flatten()

    J_G = np.asarray(jac_G_fn(z))
    if J_G.ndim == 1:
        J_G = J_G.reshape(1, -1)

    J_H = np.asarray(jac_H_fn(z))
    if J_H.ndim == 1:
        J_H = J_H.reshape(1, -1)

    J_g = None
    if jac_g_fn is not None:
        J_g = np.asarray(jac_g_fn(z))
        if J_g.ndim == 1:
            J_g = J_g.reshape(1, -1)

    return grad_f, J_g, J_G, J_H


def _classify_complementarity_indices(z: np.ndarray, problem: ProblemSpec, tol: float = _ACTIVE_TOL):
    # Classify complementarity indices into active sets.
    from mpecss.helpers.loaders.macmpec_loader import evaluate_GH
    G, H = evaluate_GH(z, problem)

    lbG_eff = np.array(problem.get('lbG_eff', np.zeros(len(G))), dtype=float)
    lbH_eff = np.array(problem.get('lbH_eff', np.zeros(len(H))), dtype=float)
    G_is_free = list(problem.get('G_is_free', []))
    if len(G_is_free) < len(G):
        G_is_free.extend([False] * (len(G) - len(G_is_free)))

    G_shifted = G.copy()
    for i in range(len(G)):
        if not G_is_free[i]:
            G_shifted[i] = G_shifted[i] - lbG_eff[i]
    H_shifted = H - lbH_eff

    I_G = []  # G_i=0, H_i strictly feasible
    I_H = []  # H_i=0, G_i strictly feasible (or free)
    I_ubH = [] # H_i=ubH_i, G_i strictly feasible (<0)
    I_B_lower = [] # G_i=0, H_i=0
    I_B_upper = [] # G_i=0, H_i=ubH_i
    I_free = []

    ubH_finite = problem.get('ubH_finite', [])
    ubH_map = {i: ub for i, ub in ubH_finite}
    for i in range(len(G)):
        g_val = G_shifted[i]
        h_val = H_shifted[i]
        
        g_active = abs(g_val) < tol
        h_lower_active = abs(h_val) < tol
        h_upper_active = False
        
        if i in ubH_map:
            if abs(H[i] - ubH_map[i]) < tol:
                h_upper_active = True

        if h_lower_active:
            if g_active:
                I_B_lower.append(i)
            else:
                I_H.append(i)
        elif h_upper_active:
            if g_active:
                I_B_upper.append(i)
            else:
                I_ubH.append(i)
        else:
            if g_active:
                I_G.append(i)
            else:
                I_free.append(i)

    return I_G, I_H, I_ubH, I_B_lower, I_B_upper, I_free


def check_mpec_licq(z: np.ndarray, problem: ProblemSpec, tol: float = _LICQ_TOL):
    # Step 1: The "Shortcut" (LICQ Check).
    grad_f, J_g, J_G, J_H = _compute_jacobians(z, problem)
    I_G, I_H, I_ubH, I_B_lower, I_B_upper, I_free = _classify_complementarity_indices(z, problem, tol=_ACTIVE_TOL)
    n_x = problem['n_x']
    
    active_rows = []
    
    if J_g is not None:
        n_con = problem.get('n_con', 0)
        info = problem['build_casadi'](0, 0)
        lbg = np.array(info['lbg'][:n_con])
        ubg = np.array(info['ubg'][:n_con])
        g_expr_orig = info['g'][:n_con]
        _g_eval_fn = ca.Function('g_licq_eval', [info['x']], [g_expr_orig])
        g_val = np.asarray(_g_eval_fn(z)).flatten()
        
        for j in range(n_con):
            if abs(g_val[j] - lbg[j]) < tol or abs(g_val[j] - ubg[j]) < tol:
                active_rows.append(J_g[j])
    
    for i in I_G:
        active_rows.append(J_G[i])
    for i in I_B_lower:
        active_rows.append(J_G[i])
        active_rows.append(J_H[i])
    for i in I_B_upper:
        active_rows.append(J_G[i])
        active_rows.append(J_H[i])
    
    for i in I_H:
        active_rows.append(J_H[i])
    for i in I_ubH:
        active_rows.append(J_H[i])
    
    n_active = len(active_rows)
    if n_active == 0:
        return True, 0, 0, 'No active constraints'
    
    A = np.vstack(active_rows)
    rank = np.linalg.matrix_rank(A, tol=tol)
    licq_holds = (rank == n_active)
    
    details = f'rank={rank}, n_active={n_active}, |I_B_lower|={len(I_B_lower)}, |I_B_upper|={len(I_B_upper)}'
    if not licq_holds:
        details += ' (rank-deficient)'
    
    return licq_holds, rank, n_active, details


def certify_bstationarity(z, problem, f_val=None, tol=_BSTAT_TOL, dir_bound=None, timeout=None):
    # B-stationarity certification via LPEC enumeration.
    cert_reason = _unsupported_certificate_reason(problem)
    if cert_reason:
        return None, float('nan'), None, {
            'lpec_status': 'unsupported_nonstandard_bounds',
            'classification': 'uncertified',
            'reason': cert_reason,
            'timed_out': False,
            'elapsed_s': 0.0,
            'n_active_G': None,
            'n_active_H': None,
            'n_branches_total': None,
            'n_branches_explored': 0,
            'n_feasible_branches': 0,
        }

    if dir_bound is None:
        dir_bound = _DIR_BOUND
    if timeout is None:
        timeout = _BSTAT_TIMEOUT

    n_x = problem['n_x']
    n_con = problem.get('n_con', 0)
    z = np.asarray(z).flatten()

    current_comp_res = float(complementarity_residual(z, problem))
    current_mcp_res = float(mcp_feasibility_residual(z, problem))
    if current_mcp_res > tol * 100:
        logger.warning(
            f"B-stat certification skipped: mcp_feas_res={current_mcp_res:.2e} >> tol={tol:.2e}"
        )
        return False, float('inf'), False, {
            'lpec_status': 'infeasible_skip',
            'classification': 'not_certified_infeasible',
            'comp_res': current_comp_res,
            'mcp_feas_res': current_mcp_res,
            'timed_out': False,
            'elapsed_s': 0.0,
            'n_active_G': None,
            'n_active_H': None,
            'n_branches_total': 0,
            'n_branches_explored': 0,
            'n_feasible_branches': 0,
        }

    grad_f, J_g, J_G, J_H = _compute_jacobians(z, problem)
    I_G, I_H, I_ubH, I_B_lower, I_B_upper, I_free = _classify_complementarity_indices(z, problem)

    licq_holds, licq_rank, n_active, licq_details = check_mpec_licq(z, problem)

    n_biactive = len(I_B_lower) + len(I_B_upper)
    logger.info(f'B-stat check: |I_G|={len(I_G)}, |I_H|={len(I_H)}, |I_ubH|={len(I_ubH)}, |I_B|={n_biactive}, LICQ={licq_holds}')

    from scipy.optimize import linprog
    t_start = time.time()

    A_ub_base = []
    b_ub_base = []
    A_eq_base = []
    b_eq_base = []

    bounds = [(-dir_bound, dir_bound) for _ in range(n_x)]

    info = problem['build_casadi'](0, 0)
    lbx = np.array(info['lbx'])
    ubx = np.array(info['ubx'])
    lbg = np.array(info['lbg']) if n_con > 0 else np.array([])
    ubg = np.array(info['ubg']) if n_con > 0 else np.array([])

    _BIG = 1e20
    for i in range(n_x):
        if lbx[i] > -_BIG:
            bounds[i] = (max(bounds[i][0], lbx[i] - z[i]), bounds[i][1])
        if ubx[i] < _BIG:
            bounds[i] = (bounds[i][0], min(bounds[i][1], ubx[i] - z[i]))

    if J_g is not None and n_con > 0:
        g_orig_expr = info['g'][:n_con]
        _g_eval_fn = ca.Function('g_bstat_eval', [info['x']], [g_orig_expr])
        g_val = np.asarray(_g_eval_fn(z)).flatten()
        for j in range(n_con):
            lb_active = abs(g_val[j] - lbg[j]) < tol if lbg[j] > -_BIG else False
            ub_active = abs(g_val[j] - ubg[j]) < tol if ubg[j] < _BIG else False
            if lb_active and ub_active:
                A_eq_base.append(J_g[j])
                b_eq_base.append(0.0)
            elif lb_active:
                A_ub_base.append(-J_g[j])
                b_ub_base.append(0.0)
            elif ub_active:
                A_ub_base.append(J_g[j])
                b_ub_base.append(0.0)

    for i in I_G:
        A_eq_base.append(J_G[i])
        b_eq_base.append(0.0)

    for i in I_H:
        A_eq_base.append(J_H[i])
        b_eq_base.append(0.0)

    for i in I_ubH:
        A_eq_base.append(J_H[i])
        b_eq_base.append(0.0)

    A_eq = np.vstack(A_eq_base) if A_eq_base else None
    b_eq = np.array(b_eq_base) if b_eq_base else None

    best_obj = 0.0
    best_direction = None
    best_branch = -1
    timed_out = False
    cap_exceeded = False
    branches_explored = 0
    n_feasible_branches = 0

    if n_biactive == 0:
        A_ub = np.vstack(A_ub_base) if A_ub_base else None
        b_ub = np.array(b_ub_base) if b_ub_base else None
        branches_explored = 1

        try:
            result = linprog(grad_f, A_ub=A_ub, b_ub=b_ub,
                            A_eq=A_eq, b_eq=b_eq,
                            bounds=bounds, method='highs')
            if result.success:
                n_feasible_branches = 1
                best_obj = result.fun
                best_direction = result.x.copy()
        except Exception as e:
            logger.debug(f'LP solve failed (no biactive): {e}')
    else:
        max_enum = 2**n_biactive
        enum_cap = 2**15

        if max_enum > enum_cap:
            cap_exceeded = True
            max_enum = enum_cap
            logger.warning(f"B-stat enumeration capped at 2^15 branches (need 2^{n_biactive})")

        for branch_idx in range(max_enum):
            branches_explored = branch_idx + 1
            if time.time() - t_start > timeout:
                timed_out = True
                break

            A_ub_branch = list(A_ub_base)
            b_ub_branch = list(b_ub_base)
            A_eq_branch = list(A_eq_base)
            b_eq_branch = list(b_eq_base)

            bit_pos = 0
            
            for i in I_B_lower:
                if (branch_idx >> bit_pos) & 1:
                    A_eq_branch.append(J_G[i])
                    b_eq_branch.append(0.0)
                    A_ub_branch.append(-J_H[i]) # >= 0
                    b_ub_branch.append(0.0)
                else:
                    A_eq_branch.append(J_H[i])
                    b_eq_branch.append(0.0)
                    A_ub_branch.append(-J_G[i]) # >= 0
                    b_ub_branch.append(0.0)
                bit_pos += 1
                
            for i in I_B_upper:
                if (branch_idx >> bit_pos) & 1:
                    A_eq_branch.append(J_G[i])
                    b_eq_branch.append(0.0)
                    A_ub_branch.append(J_H[i]) # <= 0
                    b_ub_branch.append(0.0)
                else:
                    A_eq_branch.append(J_H[i])
                    b_eq_branch.append(0.0)
                    A_ub_branch.append(J_G[i]) # <= 0
                    b_ub_branch.append(0.0)
                bit_pos += 1

            A_ub = np.vstack(A_ub_branch) if A_ub_branch else None
            b_ub = np.array(b_ub_branch) if b_ub_branch else None
            A_eq = np.vstack(A_eq_branch) if A_eq_branch else None
            b_eq = np.array(b_eq_branch) if b_eq_branch else None

            try:
                result = linprog(grad_f, A_ub=A_ub, b_ub=b_ub,
                                A_eq=A_eq, b_eq=b_eq,
                                bounds=bounds, method='highs')
                if result.success:
                    n_feasible_branches += 1
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_direction = result.x.copy()
                        best_branch = branch_idx
            except Exception as e:
                logger.debug(f'LP solve failed for branch {branch_idx}: {e}')
                continue

    if timed_out or cap_exceeded:
        if best_obj >= -tol:
            is_bstat = None
            classification = 'uncertified_favorable'
        else:
            is_bstat = False
            classification = 'not B-stationary'
        lpec_status = 'timed_out' if timed_out else 'cap_exceeded'
    else:
        is_bstat = best_obj >= -tol
        classification = 'B-stationary' if is_bstat else 'not B-stationary'
        lpec_status = 'complete'

    details = {
        'lpec_status': lpec_status,
        'licq_rank': licq_rank,
        'n_biactive': n_biactive,
        'n_I_G': len(I_G),
        'n_I_H': len(I_H),
        'n_active_G': len(I_G),
        'n_active_H': len(I_H),
        'n_eq_constraints': len(A_eq_base),
        'n_ub_constraints': len(A_ub_base),
        'classification': classification,
        'best_direction': best_direction,
        'best_branch_idx': best_branch,
        'branches_enumerated': branches_explored,
        'n_branches_total': 1 if n_biactive == 0 else 2**n_biactive,
        'n_branches_explored': branches_explored,
        'n_feasible_branches': n_feasible_branches,
        'timed_out': timed_out,
        'elapsed_s': time.time() - t_start,
        'lpec_obj': best_obj,
        'descent_found_before_termination': bool((timed_out or cap_exceeded) and best_obj < -tol),
    }

    logger.info(f'B-stat result: obj={best_obj:.2e}, is_bstat={is_bstat}, status={lpec_status}')
    return is_bstat, best_obj, licq_holds, details


def bstat_post_check(result, problem, timeout=None, eps_tol=1e-6):
    # Convenience wrapper: run B-stationarity check on MPECSS result.
    result = dict(result)

    status = result.get('status')
    stationarity = result.get('stationarity')
    comp_res = result.get('comp_res', float('inf'))

    _non_converged_statuses = ('comp_infeasible', 'nlp_failure', 'stationarity_unverifiable', 
                               'restoration_stagnation', 'max_restorations', 'stagnation', 'max_iter')
    should_check = (
        (status == 'converged' and stationarity in ('S', 'C')) or
        (status in _non_converged_statuses and comp_res <= eps_tol * 10)  # Within 10x tolerance
    )

    if not should_check:
        if result.get('b_stationarity') is None:
            result['b_stationarity'] = None
        if result.get('lpec_obj') is None:
            result['lpec_obj'] = None
        if result.get('licq_holds') is None:
            result['licq_holds'] = None
        if not result.get('bstat_details'):
            result['bstat_details'] = None
        logger.info(f"Skipping B-stat check: status={status}, stationarity={stationarity}, comp_res={comp_res:.3e}")
        return result
    
    z = result['z_final']
    f_val = result.get('f_final')
    
    try:
        is_bstat, lpec_obj, licq_holds, details = certify_bstationarity(z, problem, f_val=f_val, timeout=timeout)
        result['b_stationarity'] = is_bstat
        result['lpec_obj'] = lpec_obj
        result['licq_holds'] = licq_holds
        result['bstat_details'] = details
        
        if is_bstat is True:
            result['stationarity'] = 'B'
            result['status'] = 'converged'
            result['sign_test_pass'] = bool(result.get('sign_test_pass'))
            logger.info('Stationarity upgraded: S → B (LPEC certified)')
        elif is_bstat is False:
            result['stationarity'] = 'C'
            result['status'] = 'converged'
            result['sign_test_pass'] = False
        else:
            result['stationarity'] = 'FAIL'
            result['status'] = 'stationarity_unverifiable'
            result['sign_test_pass'] = False
    except Exception as e:
        logger.warning(f'B-stat check failed: {e}')
        result['b_stationarity'] = None
        result['lpec_obj'] = None
        result['licq_holds'] = None
        result['bstat_details'] = {'error': str(e), 'lpec_status': 'exception'}
        result['stationarity'] = 'FAIL'
        result['status'] = 'stationarity_unverifiable'
        result['sign_test_pass'] = False
    
    return result
