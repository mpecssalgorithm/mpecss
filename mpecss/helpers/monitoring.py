# Monitoring and Adaptive Timeout for MPECSS Phases.

import logging
import time
from typing import Optional, Any, Tuple, Dict

from mpecss.helpers.monitoring_system import (
    log_peak_memory,
    log_gpu_memory,
    check_gpu_available,
    get_system_info,
)
from mpecss.helpers.monitoring_timeout import PhaseTimeout, run_phase_with_timeout

logger = logging.getLogger('mpecss.monitoring')

MAX_PHASE3_BRANCHES_CPU = 2**15  # 32,768 branches


def adaptive_branch_cap(n_biactive: int, gpu_available: bool = False) -> Tuple[int, str]:
    # Determine the maximum branch enumeration for Phase III.
    total_branches = 2 ** n_biactive

    if total_branches <= MAX_PHASE3_BRANCHES_CPU:
        return total_branches, 'full'

    if gpu_available:
        return min(total_branches, 2**20), 'gpu_batched'  # 1M branches max

    logger.warning(
        f"Phase III: {total_branches} branches exceeds CPU budget ({MAX_PHASE3_BRANCHES_CPU}). "
        f"Capping enumeration. Consider GPU acceleration for complete certification."
    )
    return MAX_PHASE3_BRANCHES_CPU, 'capped'



class PhaseTimer:
    # Context manager for timing phases with memory logging.

    def __init__(self, name: str = "Phase"):
        self.name = name
        self.start_time = 0.0
        self.elapsed = 0.0
        self.start_ram_mb = 0.0
        self.peak_ram_mb = 0.0
        self.gpu_mem_mb: Optional[float] = None

    def __enter__(self):
        self.start_ram_mb = log_peak_memory()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time
        self.peak_ram_mb = log_peak_memory()
        self.gpu_mem_mb = log_gpu_memory()

        logger.info(
            f"{self.name}: {self.elapsed:.2f}s, "
            f"RAM: {self.peak_ram_mb:.0f} MB"
            + (f", GPU: {self.gpu_mem_mb:.0f} MB" if self.gpu_mem_mb else "")
        )

    def to_dict(self) -> Dict[str, Any]:
        # Convert timing info to dictionary for CSV export.
        return {
            f"{self.name}_time_s": self.elapsed,
            f"{self.name}_ram_mb": self.peak_ram_mb,
            f"{self.name}_gpu_mb": self.gpu_mem_mb,
        }
