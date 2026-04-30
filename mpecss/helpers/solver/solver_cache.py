"""Caching utilities for repeated solver calls."""

import logging
import gc as garbage_collector
from typing import Dict, Any, Optional

import casadi as ca

from mpecss.helpers.solver.solver_cache_keys import _t_round, _tol_bucket, _cache_key
from mpecss.helpers.solver.solver_cache_store import (
    LRUCache,
    MAX_TEMPLATE_CACHE_SIZE,
    MAX_SOLVER_CACHE_SIZE,
    MAX_PARAMETRIC_CACHE_SIZE,
    MAX_INFO_CACHE_SIZE,
    USE_WEAK_REFS_FOR_SOLVERS,
    _TEMPLATE_CACHE,
    _SOLVER_CACHE,
    _INFO_CACHE,
    _PARAMETRIC_CACHE,
)

logger = logging.getLogger('mpecss.solver.cache')

MEMORY_THRESHOLD_MB = 8000  # 8GB - adjust based on your system
_memory_checkpoints = []


def get_process_memory_mb() -> float:
    # Get current process memory usage in MB.
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except ImportError:
            return 0.0


def check_memory_pressure() -> bool:
    # Check if memory usage exceeds threshold.
    current_mb = get_process_memory_mb()
    _memory_checkpoints.append(current_mb)

    if len(_memory_checkpoints) > 100:
        _memory_checkpoints.pop(0)

    if current_mb > MEMORY_THRESHOLD_MB:
        logger.warning(
            f"Memory pressure detected: {current_mb:.0f} MB > {MEMORY_THRESHOLD_MB} MB threshold. "
            f"Triggering aggressive cache cleanup."
        )
        return True
    return False


def clear_solver_cache(aggressive: bool = False):
    # Clear all caches and run GC. Call between problems to free memory.
    stats = get_cache_stats()
    logger.debug(
        f"Clearing caches. Before: template={stats['template']['size']}, "
        f"solver={stats['solver']['size']}, parametric={stats['parametric']['size']}"
    )

    _SOLVER_CACHE.clear()
    _INFO_CACHE.clear()

    if aggressive:
        _TEMPLATE_CACHE.clear()
        _PARAMETRIC_CACHE.clear()
        for _ in range(3):
            garbage_collector.collect()
        logger.info("Aggressive cache cleanup completed")
    else:
        _PARAMETRIC_CACHE.clear()
        garbage_collector.collect()

    if check_memory_pressure():
        logger.warning("Memory still high after cleanup. Consider increasing limits or reducing problem batch size.")


clear_all_caches = clear_solver_cache


def get_cache_stats() -> Dict[str, Dict[str, Any]]:
    # Get statistics for all caches. Useful for monitoring memory usage.
    return {
        'template': _TEMPLATE_CACHE.stats(),
        'solver': _SOLVER_CACHE.stats(),
        'info': _INFO_CACHE.stats(),
        'parametric': _PARAMETRIC_CACHE.stats(),
        'memory_mb': get_process_memory_mb(),
        'memory_threshold_mb': MEMORY_THRESHOLD_MB,
    }


def log_cache_stats():
    # Log current cache statistics for debugging.
    stats = get_cache_stats()
    logger.info(
        f"Cache stats - Memory: {stats['memory_mb']:.0f}MB | "
        f"Template: {stats['template']['size']}/{stats['template']['max_size']} "
        f"(hit:{stats['template']['hit_rate_pct']:.0f}%) | "
        f"Solver: {stats['solver']['size']}/{stats['solver']['max_size']} "
        f"(evict:{stats['solver']['evictions']}) | "
        f"Parametric: {stats['parametric']['size']}/{stats['parametric']['max_size']}"
    )


def set_cache_limits(
    template_size: Optional[int] = None,
    solver_size: Optional[int] = None,
    parametric_size: Optional[int] = None,
    info_size: Optional[int] = None,
    memory_threshold_mb: Optional[float] = None,
):
    # Dynamically adjust cache size limits.
    global MEMORY_THRESHOLD_MB

    if template_size is not None:
        _TEMPLATE_CACHE._max_size = template_size
    if solver_size is not None:
        _SOLVER_CACHE._max_size = solver_size
    if parametric_size is not None:
        _PARAMETRIC_CACHE._max_size = parametric_size
    if info_size is not None:
        _INFO_CACHE._max_size = info_size
    if memory_threshold_mb is not None:
        MEMORY_THRESHOLD_MB = memory_threshold_mb

    logger.info(
        f"Cache limits updated: template={_TEMPLATE_CACHE._max_size}, "
        f"solver={_SOLVER_CACHE._max_size}, parametric={_PARAMETRIC_CACHE._max_size}, "
        f"memory_threshold={MEMORY_THRESHOLD_MB}MB"
    )


def _evict_problem_from_cache(prob_name):
    # Remove all concrete and parametric solver entries for prob_name.
    for cache in (_SOLVER_CACHE, _PARAMETRIC_CACHE):
        keys_to_remove = [k for k in list(cache.keys()) if k.startswith(f'{prob_name}|')]
        for k in keys_to_remove:
            if k in cache._cache:
                del cache._cache[k]


def _get_template(problem, smoothing='product'):
    # Step 1: "The Master Blueprint."
    prob_name = problem.get('name', 'unknown')
    n_x = problem.get('n_x', 0)
    n_comp = problem.get('n_comp', 0)
    n_con = problem.get('n_con', 0)
    family = problem.get('family', 'unknown')
    ckey = f'{prob_name}|{family}|{n_x}|{n_comp}|{n_con}|{smoothing}'

    cached = _TEMPLATE_CACHE.get(ckey)
    if cached is not None:
        return cached

    _sym = ca.MX.sym if n_x >= 500 else ca.SX.sym
    t_sym = _sym('t_param')
    d_sym = _sym('d_param')
    info_sym = problem['build_casadi'](t_sym, d_sym, smoothing=smoothing)
    info_sym['t_sym'] = t_sym
    info_sym['d_sym'] = d_sym

    template = (t_sym, d_sym, info_sym)
    _TEMPLATE_CACHE.put(ckey, template)

    if check_memory_pressure():
        clear_solver_cache(aggressive=True)

    return template


def build_problem(problem, t_k, delta_k, smoothing='product'):
    # Return the info dict (bounds + CasADi expressions) for problem.
    return problem['build_casadi'](t_k, delta_k, smoothing=smoothing)
