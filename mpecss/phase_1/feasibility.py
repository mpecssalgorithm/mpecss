# Phase I: Finding a starting point that actually "works."

import hashlib
import logging
import time
from typing import Any, Dict, Optional
import numpy as np
import numpy.typing as npt

from mpecss.phase_1.feasibility_nlp import _BIG, _interior_push, _solve_phase_i_nlp

logger = logging.getLogger('mpecss.feasibility')

def run_feasibility_phase(
    problem: Dict[str, Any],
    z0: npt.ArrayLike,
    solver_opts: Optional[Dict[str, Any]],
    max_attempts: int,
    n_random_restarts: int,
    seed: int = 0,
) -> Dict[str, Any]:
    # Step 1: The "Scout Mission" (Feasibility Search).
    from mpecss.helpers.comp_residuals import complementarity_residual

    n_x   = problem['n_x']
    n_comp = problem.get('n_comp', 1)
    z0    = np.asarray(z0).flatten()

    if n_x > 3000:
        logger.info(f'Phase I: skipped for large problem (n_x={n_x})')
        initial_comp_res = complementarity_residual(z0, problem)
        return {
            'z_feasible':              z0.copy(),
            'comp_res':                initial_comp_res,
            'cpu_time':                0.0,
            'obj_val':                 float('inf'),
            'solver_status':           'skipped_large',
            'n_attempts':              0,
            'n_x':                     n_x,
            'n_comp':                  n_comp,
            'initial_comp_res':        initial_comp_res,
            'final_comp_res':          initial_comp_res,
            'residual_improvement_pct': 0.0,
            'best_attempt_idx':        -1,
            'best_obj_regime':         -1,
            'n_restarts_attempted':    0,
            'n_restarts_rejected':     0,
            'best_restart_idx':        -1,
            'multistart_improved':     False,
            'ipopt_iter_count':        0,
            'displacement_from_z0':    0.0,
            'unbounded_dims_count':    0,
            'success':                 False,
            'feasibility_achieved':    False,
            'near_feasibility':        False,
        }

    t_start         = time.perf_counter()
    initial_comp_res = complementarity_residual(z0, problem)

    _lbx_arr = np.array(problem.get('lbx', [-_BIG] * n_x), dtype=float)
    _ubx_arr = np.array(problem.get('ubx', [ _BIG] * n_x), dtype=float)

    _unbounded_dims = int(np.sum((_lbx_arr <= -1e10) | (_ubx_arr >= 1e10)))

    z0 = _interior_push(z0, _lbx_arr, _ubx_arr, frac=0.1)

    best_z          = z0.copy()
    best_comp       = initial_comp_res
    best_status     = 'not_run'
    best_obj        = float('inf')
    best_attempt_idx = -1
    best_obj_regime  = -1
    attempts_used    = 0
    ipopt_total      = 0

    attempt_residuals  = {}
    attempt_objectives = {}

    _z0_scale       = max(1.0, float(np.linalg.norm(z0)))
    _MAX_DISPLACEMENT = 50.0

    n_restarts_attempted = 0
    n_restarts_rejected  = 0
    best_restart_idx     = -1
    restart_residuals    = {}
    multistart_improved  = False

    _warmstart_z = z0.copy()
    _warmstart_comp = initial_comp_res

    for attempt in range(max_attempts):
        attempts_used += 1
        try:
            z_result, obj_val, status, iter_count = _solve_phase_i_nlp(
                problem,
                z0 if attempt == 0 else _warmstart_z,
                attempt=attempt,
                solver_opts=solver_opts,
            )
            ipopt_total += iter_count
            comp_res = complementarity_residual(z_result, problem)
            attempt_residuals[attempt]  = comp_res
            attempt_objectives[attempt] = obj_val

            _disp = float(np.linalg.norm(z_result - z0)) / _z0_scale

            if comp_res < _warmstart_comp:
                _warmstart_z = z_result.copy()
                _warmstart_comp = comp_res

            _excellent_comp_res = comp_res < 1e-6
            if comp_res < best_comp and (_disp < _MAX_DISPLACEMENT or _excellent_comp_res):
                best_z           = z_result
                best_comp        = comp_res
                best_status      = status
                best_obj         = obj_val
                best_attempt_idx = attempt
                best_obj_regime  = attempt
            elif comp_res < best_comp:
                logger.debug(
                    f'Phase I attempt {attempt + 1}: comp_res={comp_res:.2e}'
                    f' rejected — displacement {_disp:.1f} > {_MAX_DISPLACEMENT}'
                )

            if comp_res < 1e-6:
                logger.info(
                    f'Phase I: feasible point found on attempt {attempt + 1}'
                    f', comp_res={comp_res:.2e}'
                )
                break
            else:
                logger.info(
                    f'Phase I: attempt {attempt + 1}, comp_res={comp_res:.2e}'
                    f' (was {initial_comp_res:.2e})'
                )

        except Exception as e:
            logger.warning(f'Phase I attempt {attempt + 1} failed: {e}')
            continue

    _COMP_GOOD_ENOUGH = 0.0001
    best_comp_before_multistart = best_comp

    if best_comp > _COMP_GOOD_ENOUGH and n_random_restarts > 0:
        _name_bytes = problem.get('name', 'anon').encode('utf-8')
        _seed_bytes = seed.to_bytes(8, 'little', signed=True)
        _combined_seed = int.from_bytes(
            hashlib.sha256(_name_bytes + _seed_bytes).digest()[:8], 'little'
        ) % (2**31 - 1)
        _rng = np.random.default_rng(_combined_seed)
        _eps = 0.001

        _lo = np.where(_lbx_arr > -1e10, _lbx_arr, best_z - 1.0)
        _hi = np.where(_ubx_arr <  1e10, _ubx_arr, best_z + 1.0)

        _lb_cand  = np.clip(_lo + _eps, _lo, _hi)
        _ub_cand  = np.clip(_hi - _eps, _lo, _hi)
        _mid_cand = 0.5 * (_lo + _hi)

        _candidates = [_lb_cand.copy(), _ub_cand.copy(), _mid_cand.copy()]

        for _sigma in (0.1, 0.3):
            _perturbed = np.asarray(
                _mid_cand + _sigma * (np.abs(_mid_cand) + 1.0) * _rng.standard_normal(n_x),
                dtype=float
            )
            _candidates.append(np.clip(_perturbed, _lo, _hi))

        for _ri, _z_cand in enumerate(_candidates[:n_random_restarts]):
            n_restarts_attempted += 1
            for _att in range(min(max_attempts, 2)):
                try:
                    _z_r, _obj_r, _s_r, _ic_r = _solve_phase_i_nlp(
                        problem,
                        _z_cand if _att == 0 else best_z,
                        attempt=_att,
                        solver_opts=solver_opts,
                    )
                    ipopt_total += _ic_r
                    _c_r   = complementarity_residual(_z_r, problem)
                    _disp_r = float(np.linalg.norm(_z_r - z0)) / _z0_scale

                    _excellent_multistart = _c_r < 1e-6
                    if _c_r < best_comp and (_disp_r < _MAX_DISPLACEMENT or _excellent_multistart):
                        best_z           = _z_r
                        best_comp        = _c_r
                        best_status      = _s_r
                        best_obj         = _obj_r
                        best_restart_idx = _ri
                        logger.info(
                            f'Phase I multistart {_ri}: comp_res '
                            f'{initial_comp_res:.2e} → {_c_r:.2e}'
                            f' (disp={_disp_r:.1f})'
                        )
                    elif _c_r < best_comp:
                        n_restarts_rejected += 1
                        logger.debug(
                            f'Phase I multistart {_ri}: comp_res={_c_r:.2e}'
                            f' rejected — displacement {_disp_r:.1f} > {_MAX_DISPLACEMENT}'
                        )

                    restart_residuals[_ri] = _c_r

                    if best_comp < _COMP_GOOD_ENOUGH:
                        break
                except Exception as _e:
                    logger.debug(f'Phase I multistart {_ri} att={_att}: {_e}')
                    continue

            if best_comp < _COMP_GOOD_ENOUGH:
                break

    cpu_time            = time.perf_counter() - t_start
    success             = best_comp < initial_comp_res
    _displacement       = float(np.linalg.norm(best_z - z0)) / _z0_scale
    _improvement_pct    = 100.0 * (initial_comp_res - best_comp) / max(initial_comp_res, 1e-10)
    _feasibility_achieved = best_comp < 1e-6
    _near_feasibility     = best_comp < 0.0001

    result = {
        'z_feasible':              np.asarray(best_z, dtype=float),
        'comp_res':                float(best_comp),
        'success':                 bool(success),
        'cpu_time':                float(cpu_time),
        'obj_val':                 float(best_obj),
        'solver_status':           str(best_status),
        'n_attempts':              int(attempts_used),
        'n_x':                     int(n_x),
        'n_comp':                  int(n_comp),
        'initial_comp_res':        float(initial_comp_res),
        'final_comp_res':          float(best_comp),
        'residual_improvement_pct': float(_improvement_pct),
        'best_attempt_idx':        int(best_attempt_idx),
        'best_obj_regime':         int(best_obj_regime),
        'attempt_0_comp_res':      float(attempt_residuals.get(0, float('inf'))),
        'attempt_1_comp_res':      float(attempt_residuals.get(1, float('inf'))),
        'attempt_2_comp_res':      float(attempt_residuals.get(2, float('inf'))),
        'n_restarts_attempted':    int(n_restarts_attempted),
        'n_restarts_rejected':     int(n_restarts_rejected),
        'best_restart_idx':        int(best_restart_idx),
        'multistart_improved':     bool(best_comp < best_comp_before_multistart),
        'ipopt_iter_count':        int(ipopt_total),
        'displacement_from_z0':    float(_displacement),
        'interior_push_frac':      0.1,
        'unbounded_dims_count':    int(_unbounded_dims),
        'feasibility_achieved':    bool(_feasibility_achieved),
        'near_feasibility':        bool(_near_feasibility),
    }

    logger.info(
        f'Phase I finished: comp_res {initial_comp_res:.2e} → {best_comp:.2e}'
        f', success={success}, time={cpu_time:.3f}s'
    )
    return result
