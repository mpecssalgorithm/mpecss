# Common MPEC Core Utilities.

import numpy as np
from typing import Any, Dict, List, Tuple, Optional

_BIG = 1e+20

_X0_PERTURBATION = 0.01

def _sanitize_bound(value: Optional[float], default: float) -> float:
    # Clip and handle None/NaN/Inf values for consistency.
    if value is None:
        return default
    v = float(value)
    if not np.isfinite(v):
        return default
    if v < -1e19:
        return -_BIG
    if v > 1e19:
        return _BIG
    return v

def _sanitize_bounds(values: Optional[List[float]], default: float) -> List[float]:
    # Apply sanitization to a list of bounds.
    if values is None:
        return []
    if isinstance(values, (int, float)):
        values = [values]
    return [_sanitize_bound(v, default) for v in values]

def evaluate_GH(x: np.ndarray, problem: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    # Evaluate complementarity functions G(x) and H(x).
    G = np.asarray(problem["G_fn"](x)).flatten()
    H = np.asarray(problem["H_fn"](x)).flatten()
    return G, H

def complementarity_residual(x: np.ndarray, problem: Dict[str, Any]) -> float:
    # Compute the canonical shifted, sign-aware MCP feasibility residual.
    from mpecss.helpers.comp_residuals import complementarity_residual as _canonical_residual

    return _canonical_residual(x, problem)

def biactive_indices(x: np.ndarray, problem: Dict[str, Any], tol: float = 1e-6) -> List[int]:
    # Return indices where both complementarity functions are near zero.
    from mpecss.helpers.comp_residuals import biactive_indices as _canonical_biactive

    return _canonical_biactive(x, problem, tol=tol)

class X0Generator:
    # Randomized initial guess generator for multistart.
    def __init__(self, w0: np.ndarray, lbx: np.ndarray, ubx: np.ndarray, perturbation: float = _X0_PERTURBATION):
        self.w0 = np.asarray(w0, dtype=float)
        self.lbx = np.asarray(lbx, dtype=float)
        self.ubx = np.asarray(ubx, dtype=float)
        self.perturbation = perturbation

    def __call__(self, seed: int = 0) -> np.ndarray:
        rng = np.random.RandomState(seed)
        x0 = self.w0.copy()
        x0 += rng.uniform(-self.perturbation, self.perturbation, size=x0.shape)
        lb = np.where(self.lbx > -1e19, self.lbx, -np.inf)
        ub = np.where(self.ubx < 1e19, self.ubx, np.inf)
        return np.clip(x0, lb + 1e-8, ub - 1e-8)
