import numpy as np


def is_solver_success(status):
    # Return True if IPOPT status indicates a successful solve.
    return status in frozenset({'Solve_Succeeded', 'Solved_To_Acceptable_Level', 'Search_Direction_Becomes_Too_Small'})


def _zero_fallback(z0, n_x, n_g):
    # Return zero-filled fallback arrays for failed solves.
    return (z0.copy(), np.zeros(n_g), np.zeros(n_x), float('inf'), np.zeros(n_g))
