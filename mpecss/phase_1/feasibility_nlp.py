import logging
from typing import Any, Dict, Optional, Tuple

import casadi as ca
import numpy as np
import numpy.typing as npt

logger = logging.getLogger('mpecss.feasibility')

_BIG: float = 1e+20

_PHASE_I_IPOPT_OPTS: Dict[str, Any] = {
    'tol':             1e-6,
    'acceptable_tol':  1e-6,
    'print_level':     0,
    'max_iter':        500,
    'mu_strategy':     'adaptive',
    'mu_oracle':       'quality-function',
    'linear_solver':   'mumps',
    'bound_push':      0.05,
    'bound_frac':      0.05,
}


def _solve_phase_i_nlp(
    problem: Dict[str, Any],
    z0: np.ndarray,
    attempt: int = 0,
    solver_opts: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, float, str, int]:
    # Step 2: "Building the Scout" (Phase I NLP).
    n_x = problem['n_x']
    n_comp = problem['n_comp']
    n_con = problem.get('n_con', 0)

    _sym = ca.MX.sym if n_x >= 500 else ca.SX.sym
    x_sym = _sym('x', n_x)

    G_fn = problem['G_fn']
    H_fn = problem['H_fn']
    G_expr = G_fn(x_sym)
    H_expr = H_fn(x_sym)

    G_is_free = list(problem.get('G_is_free', [False] * n_comp))
    if len(G_is_free) < n_comp:
        G_is_free.extend([False] * (n_comp - len(G_is_free)))
    ubH_finite = problem.get('ubH_finite', [])   # [(idx, ubH_val), ...]
    ubH_map = {i: ub for i, ub in ubH_finite}  # fast lookup

    g_parts = []
    lbg_parts = []
    ubg_parts = []

    if n_con > 0:
        info_ref = problem['build_casadi'](1.0, 0.0, smoothing='product')
        n_orig = info_ref.get('n_orig_con', n_con)
        if n_orig > 0:
            g_orig_expr = info_ref['g'][:n_orig]
            g_fn_orig = ca.Function('g_orig', [info_ref['x']], [g_orig_expr])
            g_parts.append(g_fn_orig(x_sym))
            lbg_parts.extend(info_ref['lbg'][:n_orig])
            ubg_parts.extend(info_ref['ubg'][:n_orig])

    lbG_eff = problem.get('lbG_eff', [0.0] * n_comp)
    bounded_G_idx = [i for i in range(n_comp) if not G_is_free[i]]
    if bounded_G_idx:
        g_parts.append(ca.vcat([G_expr[i] for i in bounded_G_idx]))
        lbg_parts.extend([lbG_eff[i] for i in bounded_G_idx])
        ubg_parts.extend([_BIG] * len(bounded_G_idx))

    lbH_eff = problem.get('lbH_eff', [0.0] * n_comp)
    g_parts.append(H_expr)
    lbg_parts.extend(lbH_eff)
    ubg_parts.extend([_BIG] * n_comp)

    if ubH_finite:
        g_parts.append(ca.vcat([ca.DM(ubH_map[i]) - H_expr[i]
                                for i in range(n_comp) if i in ubH_map]))
        lbg_parts.extend([0.0] * len(ubH_finite))
        ubg_parts.extend([_BIG] * len(ubH_finite))

    _empty = (ca.MX(0, 1) if n_x >= 500 else ca.SX(0, 1))
    g_sym = ca.vertcat(*g_parts) if g_parts else _empty

    lbx = problem.get('lbx', [-_BIG] * n_x)
    ubx = problem.get('ubx', [_BIG] * n_x)

    def _shifted_comp_exprs(G, H):
        # Return G/H in the same shifted coordinates used by the NLP model.
        G_parts = [
            G[i] if G_is_free[i] else G[i] - ca.DM(lbG_eff[i])
            for i in range(n_comp)
        ]
        H_parts = [H[i] - ca.DM(lbH_eff[i]) for i in range(n_comp)]
        return ca.vcat(G_parts), ca.vcat(H_parts)

    def _smooth_pos(v):
        # Smooth approximation of max(v, 0) for IPOPT-friendly objectives.
        eps_smooth = 1e-12
        return 0.5 * (v + ca.sqrt(v**2 + eps_smooth))

    def _make_violation_terms(G, H):
        G_shifted, H_shifted = _shifted_comp_exprs(G, H)
        terms = []
        for i in range(n_comp):
            if i in ubH_map:
                upper_slack = ca.DM(ubH_map[i]) - H[i]
                terms.append(_smooth_pos(G_shifted[i] * H_shifted[i]))
                terms.append(_smooth_pos((-G_shifted[i]) * upper_slack))
                if abs(float(ubH_map[i]) - float(lbH_eff[i])) <= 1e-12:
                    terms.append(G_shifted[i])
                    terms.append(-G_shifted[i])
            else:
                terms.append(G_shifted[i] * H_shifted[i])
                if G_is_free[i]:
                    terms.append(_smooth_pos(-G_shifted[i]))
        return ca.vertcat(*terms)

    if attempt == 0:
        products = _make_violation_terms(G_expr, H_expr)
        f_sym = ca.sumsqr(products)

        nlp = {'x': x_sym, 'f': f_sym, 'g': g_sym}
        opts = dict(_PHASE_I_IPOPT_OPTS)
        if solver_opts:
            opts.update(solver_opts)
        from mpecss.helpers.solver_wrapper import build_universal_nlp_solver
        solver = build_universal_nlp_solver('phase_i', n_x, nlp, ipopt_opts=opts)
        sol = solver(x0=z0, lbg=lbg_parts, ubg=ubg_parts, lbx=lbx, ubx=ubx)

    elif attempt == 1:
        products = _make_violation_terms(G_expr, H_expr)
        eps_smooth = 1e-10
        f_sym = ca.sum1(ca.sqrt(products**2 + eps_smooth))

        nlp = {'x': x_sym, 'f': f_sym, 'g': g_sym}
        opts = dict(_PHASE_I_IPOPT_OPTS)
        if solver_opts:
            opts.update(solver_opts)
        from mpecss.helpers.solver_wrapper import build_universal_nlp_solver
        solver = build_universal_nlp_solver('phase_i', n_x, nlp, ipopt_opts=opts)
        sol = solver(x0=z0, lbg=lbg_parts, ubg=ubg_parts, lbx=lbx, ubx=ubx)

    else:
        x_aug = _sym('x_aug', n_x + 1)
        x_orig = x_aug[:n_x]
        t_epi = x_aug[n_x]

        G_aug = G_fn(x_orig)
        H_aug = H_fn(x_orig)

        g_aug_parts = []
        lbg_aug = []
        ubg_aug = []

        if n_con > 0:
            info_ref = problem['build_casadi'](1.0, 0.0, smoothing='product')
            n_orig = info_ref.get('n_orig_con', n_con)
            if n_orig > 0:
                g_fn_orig = ca.Function('g_orig', [info_ref['x']],
                                        [info_ref['g'][:n_orig]])
                g_aug_parts.append(g_fn_orig(x_orig))
                lbg_aug.extend(info_ref['lbg'][:n_orig])
                ubg_aug.extend(info_ref['ubg'][:n_orig])

        if bounded_G_idx:
            g_aug_parts.append(ca.vcat([G_aug[i] for i in bounded_G_idx]))
            lbg_aug.extend([lbG_eff[i] for i in bounded_G_idx])
            ubg_aug.extend([_BIG] * len(bounded_G_idx))

        g_aug_parts.append(H_aug)
        lbg_aug.extend(lbH_eff)
        ubg_aug.extend([_BIG] * n_comp)

        if ubH_finite:
            g_aug_parts.append(ca.vcat([
                ca.DM(ubH_map[i]) - H_aug[i]
                for i in range(n_comp) if i in ubH_map
            ]))
            lbg_aug.extend([0.0] * len(ubH_finite))
            ubg_aug.extend([_BIG] * len(ubH_finite))

        G_shifted_aug, H_shifted_aug = _shifted_comp_exprs(G_aug, H_aug)

        for i in range(n_comp):
            gi = G_shifted_aug[i]
            hi = H_shifted_aug[i]
            g_aug_parts.append((gi * hi) - t_epi)
            lbg_aug.append(-_BIG)
            ubg_aug.append(0.0)

            if G_is_free[i] and i not in ubH_map:
                g_aug_parts.append((-gi) - t_epi)
                lbg_aug.append(-_BIG)
                ubg_aug.append(0.0)

        for i, ub in ubH_finite:
            gi = G_shifted_aug[i]
            hi = H_aug[i]
            upper_slack = ca.DM(ub) - hi
            g_aug_parts.append(((-gi) * upper_slack) - t_epi)
            lbg_aug.append(-_BIG)
            ubg_aug.append(0.0)
            if abs(float(ub) - float(lbH_eff[i])) <= 1e-12:
                g_aug_parts.append(gi - t_epi)
                lbg_aug.append(-_BIG)
                ubg_aug.append(0.0)
                g_aug_parts.append((-gi) - t_epi)
                lbg_aug.append(-_BIG)
                ubg_aug.append(0.0)

        g_aug_sym = ca.vertcat(*g_aug_parts)

        lbx_aug = list(problem.get('lbx', [-_BIG] * n_x)) + [0.0]
        ubx_aug = list(problem.get('ubx', [_BIG] * n_x)) + [_BIG]
        z0_aug = np.append(z0, 1.0)

        nlp = {'x': x_aug, 'f': t_epi, 'g': g_aug_sym}
        opts = dict(_PHASE_I_IPOPT_OPTS)
        if solver_opts:
            opts.update(solver_opts)
        from mpecss.helpers.solver_wrapper import build_universal_nlp_solver
        solver = build_universal_nlp_solver('phase_i_epi', n_x, nlp, ipopt_opts=opts)
        sol = solver(x0=z0_aug, lbg=lbg_aug, ubg=ubg_aug, lbx=lbx_aug, ubx=ubx_aug)

        z_result = np.asarray(sol['x'], dtype=float).flatten()[:n_x]
        obj_val = float(sol['f'])
        stats = dict(solver.stats())
        status = str(stats.get('return_status', 'unknown'))
        iter_count = int(stats.get('iter_count', 0))
        return z_result, obj_val, status, iter_count

    z_result = np.asarray(sol['x'], dtype=float).flatten()
    obj_val = float(sol['f'])
    stats = dict(solver.stats())
    status = str(stats.get('return_status', 'unknown'))
    iter_count = int(stats.get('iter_count', 0))

    return z_result, obj_val, status, iter_count


def _interior_push(z: npt.ArrayLike, lbx: npt.ArrayLike, ubx: npt.ArrayLike,
                   frac: float = 0.1) -> np.ndarray:
    # Push z into the strict interior of [lbx, ubx] by fraction frac.
    z = np.array(z, dtype=float)
    lbx = np.asarray(lbx, dtype=float)
    ubx = np.asarray(ubx, dtype=float)

    n = len(z)
    for i in range(n):
        lb = lbx[i] if lbx[i] > -1e15 else None
        ub = ubx[i] if ubx[i] < 1e15 else None

        if lb is not None and ub is not None:
            gap = ub - lb
            if gap <= 1e-10:
                continue
            lo_safe = lb + frac * gap
            hi_safe = ub - frac * gap
            z[i] = float(np.clip(z[i], lo_safe, hi_safe))

        elif lb is not None:
            push = frac * max(1.0, abs(lb))
            z[i] = max(z[i], lb + push)

        elif ub is not None:
            push = frac * max(1.0, abs(ub))
            z[i] = min(z[i], ub - push)

    return z
