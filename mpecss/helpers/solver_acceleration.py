# The "Smart Selector": Choosing the best tools for the job.

import logging

logger = logging.getLogger('mpecss.solver.acceleration')

SQP_SIZE_THRESHOLD = 400


def select_linear_solver_oss(n_x: int) -> str:
    # Choosing the "Heavy Lifter" (Linear Solver).
    return "mumps"


def select_nlp_solver(n_x: int) -> str:
    # Picking the Right Engine (NLP Solver).
    try:
        from mpecss.helpers.solver_sqp import QPOASES_AVAILABLE
        if QPOASES_AVAILABLE and n_x <= SQP_SIZE_THRESHOLD:
            logger.debug(f"Selecting SQP+qpOASES for n_x={n_x} (≤{SQP_SIZE_THRESHOLD})")
            return 'sqp'
    except ImportError:
        pass
    
    logger.debug(f"Selecting IPOPT+MUMPS for n_x={n_x}")
    return 'ipopt'


def is_sqp_recommended(n_x: int) -> bool:
    # Check if SQP+qpOASES is recommended for a problem of given size.
    try:
        from mpecss.helpers.solver_sqp import QPOASES_AVAILABLE
        return QPOASES_AVAILABLE and n_x <= SQP_SIZE_THRESHOLD
    except ImportError:
        return False
