import logging

import casadi as ca

logger = logging.getLogger('mpecss.solver.sqp')

DEFAULT_SQP_OPTS = {
    'max_iter': 100,              # Maximum SQP iterations
    'tol_opt': 1e-8,              # Optimality tolerance (KKT residual)
    'tol_feas': 1e-8,             # Feasibility tolerance
    'tol_step': 1e-12,            # Minimum step size
    'hessian_approximation': 'bfgs',  # 'bfgs', 'gauss-newton', or 'exact'
    'merit_function': 'l1',       # 'l1' penalty or 'filter'
    'line_search': True,          # Enable Armijo line search
    'armijo_c1': 1e-4,            # Armijo sufficient decrease constant
    'max_ls_iter': 20,            # Maximum line search iterations
    'regularization': 1e-8,       # Hessian regularization for convexity
    'print_level': 0,             # 0=silent, 1=summary, 2=detailed
}

DEFAULT_QPOASES_OPTS = {
    'printLevel': 'none',         # Suppress qpOASES output
    'enableRegularisation': True,
    'enableEqualities': True,
    'terminationTolerance': 1e-12,
    'boundTolerance': 1e-12,
    'verbose': False,             # Suppress verbose output
}

SQP_SIZE_THRESHOLD = 400


def _check_qpoases_available():
    # Check if qpOASES is available in CasADi.
    try:
        h = ca.DM.eye(2)
        a = ca.DM.ones(1, 2)  # One constraint row
        qp = {'h': h.sparsity(), 'a': a.sparsity()}
        solver = ca.conic('test', 'qpoases', qp, {'printLevel': 'none'})
        solver(h=h, g=ca.DM.zeros(2), a=a, lba=-1, uba=1, lbx=-10, ubx=10)
        return True
    except Exception as e:
        logger.warning(f"qpOASES not available: {e}")
        return False


QPOASES_AVAILABLE = _check_qpoases_available()
