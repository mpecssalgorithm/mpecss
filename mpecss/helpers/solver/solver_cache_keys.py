import math
import sys


def _t_round(t):
    # Round t/delta to 4 significant figures for stable cache keys.
    if t == 0:
        return 0
    mag = math.floor(math.log10(abs(t)))
    return round(t, -mag + 3)


def _tol_bucket(tol):
    # Round IPOPT tol to the nearest power of 10 for cache key stability.
    if tol <= 0:
        return 1e-08
    exp = math.floor(math.log10(tol + sys.float_info.min))
    return 10 ** exp


def _cache_key(problem_name, n_x, tol_bucket):
    # Composite cache key (retained for external callers).
    return f'{problem_name}|{n_x}|{tol_bucket}'
