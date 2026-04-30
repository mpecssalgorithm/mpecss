import logging
import signal
import sys
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger('mpecss.monitoring')


class PhaseTimeout(Exception):
    # Raised when a phase exceeds its wall-clock budget.
    pass


def timeout_handler(signum, frame):
    # Signal handler for SIGALRM timeout.
    raise PhaseTimeout("Phase exceeded wall-clock budget")


def run_phase_with_timeout(
    phase_fn: Callable,
    args: tuple,
    kwargs: Optional[dict] = None,
    wall_budget_seconds: float = 120.0,
    phase_name: str = "unknown"
) -> Tuple[Any, str]:
    # Run a phase function with a hard wall-clock limit.
    if kwargs is None:
        kwargs = {}

    if sys.platform == 'win32':
        return _run_with_timeout_threading(phase_fn, args, kwargs, wall_budget_seconds, phase_name)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(wall_budget_seconds))

    try:
        result = phase_fn(*args, **kwargs)
        signal.alarm(0)
        return result, "completed"
    except PhaseTimeout:
        logger.warning(f"{phase_name}: Timeout after {wall_budget_seconds}s")
        return None, "timeout"
    finally:
        signal.alarm(0)


def _run_with_timeout_threading(
    phase_fn: Callable,
    args: tuple,
    kwargs: dict,
    wall_budget_seconds: float,
    phase_name: str
) -> Tuple[Any, str]:
    # Use a spawned process so wall-clock timeouts work consistently on Windows.
    import multiprocessing

    ctx = multiprocessing.get_context('spawn')
    result_queue = ctx.Queue()

    def _worker_proc(q, fn, a, kw):
        # Worker executed in the child process.
        try:
            r = fn(*a, **kw)
            q.put(('result', r))
        except Exception as exc:
            q.put(('error', exc))

    try:
        proc = ctx.Process(
            target=_worker_proc,
            args=(result_queue, phase_fn, args, kwargs),
            daemon=True,  # auto-cleaned up if parent dies
        )
        proc.start()
    except Exception as pickle_err:
        logger.warning(
            f"{phase_name}: Cannot use process-based timeout "
            f"(not picklable: {pickle_err}). "
            f"Falling back to thread-based timeout — IPOPT may not be killable on Windows. "
            f"Wrap your function or use benchmark_utils multiprocessing path instead."
        )
        return _run_with_timeout_thread_fallback(phase_fn, args, kwargs, wall_budget_seconds, phase_name)

    proc.join(timeout=wall_budget_seconds)

    if proc.is_alive():
        logger.warning(
            f"{phase_name}: Timeout after {wall_budget_seconds}s — "
            f"forcibly terminating worker process (PID={proc.pid})"
        )
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            logger.warning(f"{phase_name}: Still alive after SIGTERM, escalating to SIGKILL")
            proc.kill()
            proc.join(timeout=2)
        return None, "timeout"

    try:
        msg_type, payload = result_queue.get_nowait()
    except Exception:
        logger.warning(f"{phase_name}: Worker ended without result (crash/OOM?)")
        return None, "timeout"

    if msg_type == 'error':
        raise payload
    return payload, "completed"


def _run_with_timeout_thread_fallback(
    phase_fn: Callable,
    args: tuple,
    kwargs: dict,
    wall_budget_seconds: float,
    phase_name: str
) -> Tuple[Any, str]:
    # Pure thread-based soft timeout (fallback only).
    import threading

    result = [None]
    status = ["timeout"]
    exception = [None]

    def target():
        try:
            result[0] = phase_fn(*args, **kwargs)
            status[0] = "completed"
        except Exception as e:
            exception[0] = e
            status[0] = "error"

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=wall_budget_seconds)

    if thread.is_alive():
        logger.warning(
            f"{phase_name}: Timeout after {wall_budget_seconds}s "
            f"(thread still running — cannot forcibly kill on Windows)"
        )
        return None, "timeout"

    if exception[0] is not None:
        raise exception[0]

    return result[0], status[0]
