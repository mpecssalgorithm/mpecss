# The "Agile Sprinter": Solving small problems with speed.

import time
import logging
import numpy as np
import casadi as ca
from mpecss.helpers.solver_metrics import combine_kkt_residuals
from mpecss.helpers.solver_sqp_options import (
    DEFAULT_SQP_OPTS,
    DEFAULT_QPOASES_OPTS,
    SQP_SIZE_THRESHOLD,
    QPOASES_AVAILABLE,
)

logger = logging.getLogger('mpecss.solver.sqp')

class SQPSolver:
    # Step 1: "The Sprinter" (SQP Solver).
    
    def __init__(self, problem, sqp_opts=None, qp_opts=None):
        # Initialize SQP solver for a given problem.
        self.problem = problem
        self.n_x = problem['n_x']
        self.n_g = problem.get('n_g', 0)
        
        self.sqp_opts = dict(DEFAULT_SQP_OPTS)
        if sqp_opts:
            self.sqp_opts.update(sqp_opts)
        
        self.qp_opts = dict(DEFAULT_QPOASES_OPTS)
        if qp_opts:
            self.qp_opts.update(qp_opts)
        
        self._build_functions()
        
        self._B = None  # Current Hessian approximation
        self._prev_x = None
        self._prev_grad = None
        
        self._qp_solver = None
    
    def _build_functions(self):
        # Build CasADi functions for objective, constraints, and derivatives.
        x_sym = ca.SX.sym('x', self.n_x)
        
        f_expr = self.problem['f_fun'](x_sym)
        g_expr = self.problem.get('g_fun', lambda x: ca.SX([]))(x_sym)
        
        if g_expr.is_empty():
            g_expr = ca.SX.zeros(0)
            self.n_g = 0
        else:
            self.n_g = g_expr.shape[0]
        
        self.f_fun = ca.Function('f', [x_sym], [f_expr])
        self.g_fun = ca.Function('g', [x_sym], [g_expr])
        
        grad_f = ca.gradient(f_expr, x_sym)
        self.grad_f_fun = ca.Function('grad_f', [x_sym], [grad_f])
        
        if self.n_g > 0:
            jac_g = ca.jacobian(g_expr, x_sym)
            self.jac_g_fun = ca.Function('jac_g', [x_sym], [jac_g])
        else:
            self.jac_g_fun = None
        
        if self.sqp_opts['hessian_approximation'] == 'exact':
            lam_sym = ca.SX.sym('lam', self.n_g) if self.n_g > 0 else ca.SX.sym('lam', 0)
            lagrangian = f_expr
            if self.n_g > 0:
                lagrangian += ca.dot(lam_sym, g_expr)
            hess_L = ca.hessian(lagrangian, x_sym)[0]
            self.hess_L_fun = ca.Function('hess_L', [x_sym, lam_sym], [hess_L])
        else:
            self.hess_L_fun = None
    
    def _get_qp_solver(self, H_sparsity, A_sparsity):
        # Get or create qpOASES solver for the QP subproblem.
        if not QPOASES_AVAILABLE:
            return None
        if self._qp_solver is not None:
            return self._qp_solver
        
        try:
            qp = {
                'h': ca.Sparsity.dense(self.n_x, self.n_x),
                'a': (
                    ca.Sparsity.dense(self.n_g, self.n_x)
                    if self.n_g > 0
                    else ca.Sparsity(0, self.n_x)
                ),
            }
            self._qp_solver = ca.conic('qp', 'qpoases', qp, self.qp_opts)
            return self._qp_solver
        except Exception as e:
            logger.warning(f"Failed to create qpOASES solver: {e}")
            self._qp_solver = None
            return None
    
    def _update_bfgs(self, x_new, grad_new):
        # Update BFGS Hessian approximation.
        if self._prev_x is None:
            self._B = np.eye(self.n_x) * self.sqp_opts['regularization']
            self._prev_x = x_new.copy()
            self._prev_grad = grad_new.copy()
            return
        
        s = x_new - self._prev_x
        y = grad_new - self._prev_grad
        
        sTy = np.dot(s, y)
        if sTy > 1e-10 * np.linalg.norm(s) * np.linalg.norm(y):
            Bs = self._B @ s
            sBs = np.dot(s, Bs)
            
            self._B = (self._B 
                       - np.outer(Bs, Bs) / max(sBs, 1e-12)
                       + np.outer(y, y) / sTy)
        
        self._prev_x = x_new.copy()
        self._prev_grad = grad_new.copy()
    
    def _get_hessian(self, x, lam_g):
        # Get Hessian approximation at current point.
        mode = self.sqp_opts['hessian_approximation']
        
        if mode == 'exact' and self.hess_L_fun is not None:
            H = np.array(self.hess_L_fun(x, lam_g)).reshape(self.n_x, self.n_x)
        elif mode == 'gauss-newton':
            if self._B is None:
                self._B = np.eye(self.n_x)
            H = self._B
        else:  # BFGS
            if self._B is None:
                self._B = np.eye(self.n_x)
            H = self._B
        
        reg = self.sqp_opts['regularization']
        H = H + reg * np.eye(self.n_x)
        
        return H
    
    def _solve_qp_subproblem(self, x_k, H, grad_f, A, g_val, lbx, ubx, lbg, ubg):
        # Step 2: "Breaking it Down" (QP Subproblem).
        lbd = lbx - x_k
        ubd = ubx - x_k
        
        if self.n_g > 0:
            lba = lbg - g_val
            uba = ubg - g_val
        else:
            lba = np.array([])
            uba = np.array([])
        
        H_dm = ca.DM(H)
        g_dm = ca.DM(grad_f)
        
        if self.n_g > 0:
            A_dm = ca.DM(A)
        else:
            A_dm = ca.DM.zeros(0, self.n_x)
        
        solver = self._get_qp_solver(H_dm.sparsity(), A_dm.sparsity())
        
        if solver is None:
            return None, None, None, 'qpOASES_unavailable'
        
        try:
            sol = solver(
                h=H_dm,
                g=g_dm,
                a=A_dm,
                lba=ca.DM(lba) if len(lba) > 0 else ca.DM(),
                uba=ca.DM(uba) if len(uba) > 0 else ca.DM(),
                lbx=ca.DM(lbd),
                ubx=ca.DM(ubd),
            )
            
            d = np.array(sol['x']).flatten()
            lam_g = np.array(sol['lam_a']).flatten() if self.n_g > 0 else np.array([])
            lam_x = np.array(sol['lam_x']).flatten()
            
            stats = solver.stats()
            if stats.get('success', True):
                return d, lam_g, lam_x, 'success'
            else:
                return d, lam_g, lam_x, 'qp_failed'
                
        except Exception as e:
            logger.debug(f"QP subproblem failed: {e}")
            return None, None, None, 'qp_exception'
    
    def _line_search(self, x_k, d, f_k, grad_f_k, g_k, lbg, ubg):
        # Armijo backtracking line search with L1 merit function.
        if not self.sqp_opts['line_search']:
            return 1.0, True
        
        c1 = self.sqp_opts['armijo_c1']
        max_iter = self.sqp_opts['max_ls_iter']
        
        def constraint_violation(g_val):
            if len(g_val) == 0:
                return 0.0
            viol_lb = np.maximum(lbg - g_val, 0)
            viol_ub = np.maximum(g_val - ubg, 0)
            return np.sum(viol_lb) + np.sum(viol_ub)
        
        cv_k = constraint_violation(g_k)
        
        mu = max(1.0, np.linalg.norm(grad_f_k))
        
        merit_k = f_k + mu * cv_k
        
        dir_deriv = np.dot(grad_f_k, d)
        
        alpha = 1.0
        for _ in range(max_iter):
            x_trial = x_k + alpha * d
            f_trial = float(self.f_fun(x_trial))
            g_trial = np.array(self.g_fun(x_trial)).flatten() if self.n_g > 0 else np.array([])
            cv_trial = constraint_violation(g_trial)
            merit_trial = f_trial + mu * cv_trial
            
            if merit_trial <= merit_k + c1 * alpha * dir_deriv:
                return alpha, True
            
            alpha *= 0.5
        
        return 1.0, False
    
    def _check_convergence(self, grad_L, g_val, lbg, ubg, d):
        # Check KKT optimality conditions.
        tol_opt = self.sqp_opts['tol_opt']
        tol_feas = self.sqp_opts['tol_feas']
        tol_step = self.sqp_opts['tol_step']
        
        opt_err = np.linalg.norm(grad_L, np.inf)
        
        if self.n_g > 0:
            feas_err = max(
                np.max(np.maximum(lbg - g_val, 0)),
                np.max(np.maximum(g_val - ubg, 0))
            )
        else:
            feas_err = 0.0
        
        step_norm = np.linalg.norm(d)
        
        converged = (opt_err <= tol_opt and feas_err <= tol_feas)
        stalled = (step_norm <= tol_step)
        
        return converged, stalled, opt_err, feas_err
    
    def solve(self, x0, lam_g0=None, lam_x0=None):
        # Step 3: "Running the Sprint" (Solving).
        t0 = time.perf_counter()
        
        lbx = np.array(self.problem.get('lbx', [-np.inf] * self.n_x)).flatten()
        ubx = np.array(self.problem.get('ubx', [np.inf] * self.n_x)).flatten()
        lbg = np.array(self.problem.get('lbg', [])).flatten()
        ubg = np.array(self.problem.get('ubg', [])).flatten()
        
        x_k = np.array(x0).flatten()
        lam_g = np.zeros(self.n_g) if lam_g0 is None else np.array(lam_g0).flatten()
        lam_x = np.zeros(self.n_x) if lam_x0 is None else np.array(lam_x0).flatten()
        
        x_k = np.clip(x_k, lbx, ubx)
        
        status = 'max_iter_reached'
        iter_count = 0
        kkt_res = float('nan')
        
        for k in range(self.sqp_opts['max_iter']):
            iter_count = k + 1
            
            f_k = float(self.f_fun(x_k))
            g_k = np.array(self.g_fun(x_k)).flatten() if self.n_g > 0 else np.array([])
            grad_f_k = np.array(self.grad_f_fun(x_k)).flatten()
            
            if self.n_g > 0:
                A_k = np.array(self.jac_g_fun(x_k))
            else:
                A_k = np.zeros((0, self.n_x))
            
            if self.sqp_opts['hessian_approximation'] == 'bfgs':
                self._update_bfgs(x_k, grad_f_k)
            
            H_k = self._get_hessian(x_k, lam_g)
            
            d, lam_g_qp, lam_x_qp, qp_status = self._solve_qp_subproblem(
                x_k, H_k, grad_f_k, A_k, g_k, lbx, ubx, lbg, ubg
            )
            
            if qp_status != 'success' or d is None:
                logger.debug(f"QP failed at iter {k}: {qp_status}")
                status = f'qp_failed_{qp_status}'
                break
            
            if lam_g_qp is not None and len(lam_g_qp) > 0:
                lam_g = lam_g_qp
            if lam_x_qp is not None:
                lam_x = lam_x_qp
            
            grad_L = grad_f_k.copy()
            if self.n_g > 0:
                grad_L += A_k.T @ lam_g
            
            converged, stalled, opt_err, feas_err = self._check_convergence(
                grad_L, g_k, lbg, ubg, d
            )
            kkt_res = combine_kkt_residuals(opt_err, feas_err)
            
            if self.sqp_opts['print_level'] >= 2:
                logger.info(f"SQP iter {k}: f={f_k:.6e}, opt={opt_err:.2e}, feas={feas_err:.2e}, |d|={np.linalg.norm(d):.2e}")
            
            if converged:
                status = 'Solve_Succeeded'
                break
            
            if stalled:
                status = 'Search_Direction_Becomes_Too_Small'
                break
            
            alpha, ls_success = self._line_search(x_k, d, f_k, grad_f_k, g_k, lbg, ubg)
            
            x_k = x_k + alpha * d
            x_k = np.clip(x_k, lbx, ubx)  # Project to bounds
        
        cpu_time = time.perf_counter() - t0
        
        f_final = float(self.f_fun(x_k))
        g_final = np.array(self.g_fun(x_k)).flatten() if self.n_g > 0 else np.array([])
        
        if self.sqp_opts['print_level'] >= 1:
            logger.info(f"SQP finished: status={status}, iter={iter_count}, f={f_final:.6e}, time={cpu_time:.3f}s")
        
        return {
            'x': x_k,
            'f': f_final,
            'g': g_final,
            'lam_g': lam_g,
            'lam_x': lam_x,
            'kkt_res': kkt_res,
            'status': status,
            'iter_count': iter_count,
            'cpu_time': cpu_time,
        }


def solve_nlp_sqp(x0, problem, sqp_opts=None, qp_opts=None, lam_g0=None, lam_x0=None):
    # Convenience function to solve an NLP using SQP+qpOASES.
    solver = SQPSolver(problem, sqp_opts, qp_opts)
    return solver.solve(x0, lam_g0, lam_x0)
