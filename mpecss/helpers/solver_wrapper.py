# The Universal Translator: Connecting our math to the computer's brain.

import casadi as ca
from mpecss.helpers.solver_cache import clear_solver_cache, build_problem
from mpecss.helpers.solver_ipopt import (
    solve_smooth_subproblem,
    is_solver_success,
    solve_with_solver_fallback,
    DEFAULT_IPOPT_OPTS,
)
from mpecss.helpers.solver_acceleration import (
    select_nlp_solver,
    select_linear_solver_oss,
    is_sqp_recommended,
)

try:
    from mpecss.helpers.solver_sqp import (
        SQPSolver,
        solve_nlp_sqp,
        QPOASES_AVAILABLE,
        SQP_SIZE_THRESHOLD,
    )
except ImportError:
    QPOASES_AVAILABLE = False
    SQP_SIZE_THRESHOLD = 400


def build_universal_nlp_solver(name, n_x, nlp, ipopt_opts=None):
    # The Universal Builder.
    if ipopt_opts is None:
        ipopt_opts = dict(DEFAULT_IPOPT_OPTS)
        
    return ca.nlpsol(name, 'ipopt', nlp, {
        'ipopt': ipopt_opts,
        'print_time': False,
        'verbose': False,  # Suppress CasADi verbose output
        'error_on_fail': False
    })
