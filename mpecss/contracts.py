from typing import Any, Dict, List, Optional, Callable, TypedDict, Union
from enum import Enum
import numpy as np


class SolverStatus(str, Enum):
    CONVERGED = "converged"
    MAX_ITER = "max_iter"
    NLP_FAILURE = "nlp_failure"
    TIMEOUT = "timeout"
    STAGNATION = "stagnation"
    STATIONARITY_UNVERIFIABLE = "stationarity_unverifiable"
    UNSUPPORTED_MODEL = "unsupported_model"
    OOM = "oom"
    CRASHED = "crashed"
    LOAD_FAILED = "load_failed"


class StationarityClass(str, Enum):
    B_STATIONARY = "B"
    C_STATIONARY = "C"
    FAIL = "FAIL"


class ProblemSpec(TypedDict, total=False):
    """Typing for the MPEC problem dictionary returned by loaders."""
    name: str
    n_x: int
    n_comp: int
    n_con: int
    n_p: int
    family: str
    x0_fn: Callable[[int], np.ndarray]
    f_fn: Callable[[np.ndarray], Union[float, np.ndarray]]
    G_fn: Callable[[np.ndarray], np.ndarray]
    H_fn: Callable[[np.ndarray], np.ndarray]
    build_casadi: Callable[..., Dict[str, Any]]
    unsupported_model_reason: Optional[str]
    lbx: List[float]
    ubx: List[float]
    lbg: List[float]
    ubg: List[float]


class SolveResult(TypedDict, total=False):
    """Typing for the standard result dictionary returned by run_mpecss."""
    z_final: np.ndarray
    f_final: float
    objective: float
    comp_res: float
    kkt_res: float
    stationarity: StationarityClass
    n_outer_iters: int
    n_restorations: int
    cpu_time: float
    logs: List[Any]
    status: Union[SolverStatus, str]
    sign_test_pass: Optional[bool]
    seed: int
    b_stationarity: Optional[bool]
    lpec_obj: Optional[float]
    licq_holds: Optional[bool]
    bstat_details: Optional[Dict[str, Any]]
    phase_i_result: Optional[Dict[str, Any]]
