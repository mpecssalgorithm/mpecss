# Complementarity Residual Metrics

import numpy as np
from typing import Any, Dict, List, Tuple


def _evaluate_GH_raw(x: np.ndarray, problem: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    # Evaluate raw G and H functions (no shifting).
    G_fn = problem.get('G_fn')
    H_fn = problem.get('H_fn')

    if G_fn is None or H_fn is None:
        return np.array([]), np.array([])

    G = np.asarray(G_fn(x)).flatten()
    H = np.asarray(H_fn(x)).flatten()

    return G, H


def _get_shifted_GH(x: np.ndarray, problem: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    # Get shifted G and H values accounting for nonzero lower bounds.
    G, H = _evaluate_GH_raw(x, problem)
    if len(G) == 0:
        return G, H

    lbG_eff = np.array(problem.get('lbG_eff', np.zeros(len(G))), dtype=float)
    lbH_eff = np.array(problem.get('lbH_eff', np.zeros(len(H))), dtype=float)
    G_is_free = np.array(problem.get('G_is_free', [False] * len(G)), dtype=bool)
    H_is_free = np.array(problem.get('H_is_free', [False] * len(H)), dtype=bool)

    G_shifted = G.copy()
    H_shifted = H.copy()
    G_shifted[~G_is_free] = G_shifted[~G_is_free] - lbG_eff[~G_is_free]
    H_shifted[~H_is_free] = H_shifted[~H_is_free] - lbH_eff[~H_is_free]

    return G_shifted, H_shifted


def _bool_metadata(problem: Dict[str, Any], key: str, n: int) -> np.ndarray:
    # Return a length-n boolean metadata vector with a conservative default.
    values = np.array(problem.get(key, [False] * n), dtype=bool)
    if len(values) < n:
        values = np.pad(values, (0, n - len(values)), constant_values=False)
    return values[:n]


def _upper_h_slacks(H_raw: np.ndarray, problem: Dict[str, Any]) -> Dict[int, float]:
    # Return upper-bound slack values in raw H coordinates.
    slacks: Dict[int, float] = {}
    for i, ub_raw in problem.get('ubH_finite', []):
        idx = int(i)
        if idx < len(H_raw):
            slacks[idx] = float(ub_raw) - float(H_raw[idx])
    return slacks


def _mcp_component_residuals(x: np.ndarray, problem: Dict[str, Any]) -> np.ndarray:
    # Return per-component shifted, sign-aware MCP feasibility residuals.
    G_shifted, H_shifted = _get_shifted_GH(x, problem)
    if len(G_shifted) == 0:
        return np.array([], dtype=float)

    G_raw, H_raw = _evaluate_GH_raw(x, problem)
    n_comp = len(G_shifted)
    H_is_free = _bool_metadata(problem, 'H_is_free', n_comp)
    lbH_eff = np.array(problem.get('lbH_eff', np.zeros(n_comp)), dtype=float)
    if len(lbH_eff) < n_comp:
        lbH_eff = np.pad(lbH_eff, (0, n_comp - len(lbH_eff)), constant_values=0.0)

    ubH_map = {int(i): float(ub) for i, ub in problem.get('ubH_finite', [])}
    residuals = np.zeros(n_comp, dtype=float)

    for i in range(n_comp):
        g_i = float(G_shifted[i])
        h_lower_slack = float(H_shifted[i])
        terms: List[float] = []

        if not H_is_free[i]:
            terms.append(max(-h_lower_slack, 0.0))

        if i in ubH_map:
            h_upper_slack = float(ubH_map[i]) - float(H_raw[i])
            terms.append(max(-h_upper_slack, 0.0))

            terms.append(max(g_i * h_lower_slack, 0.0))
            terms.append(max((-g_i) * h_upper_slack, 0.0))

            if abs(float(ubH_map[i]) - float(lbH_eff[i])) <= 1e-12:
                terms.append(abs(g_i))
        else:
            terms.append(max(-g_i, 0.0))
            terms.append(abs(g_i * h_lower_slack))

        residuals[i] = max(terms) if terms else 0.0

    return residuals


def mcp_feasibility_residual(x: np.ndarray, problem: Dict[str, Any]) -> float:
    # Full shifted MCP feasibility residual.
    residuals = _mcp_component_residuals(x, problem)
    if len(residuals) == 0:
        return 0.0
    return float(np.max(residuals))


def homotopy_comp_res(x: np.ndarray, problem: Dict[str, Any]) -> float:
    # Bound-aware complementarity residual for Scholtes homotopy stopping.
    return mcp_feasibility_residual(x, problem)


def biactive_residual(x: np.ndarray, problem: Dict[str, Any]) -> float:
    # Min-abs complementarity residual for stationarity set detection.
    G_shifted, H_shifted = _get_shifted_GH(x, problem)
    if len(G_shifted) == 0:
        return 0.0

    G_raw, H_raw = _evaluate_GH_raw(x, problem)
    ubH_slacks = _upper_h_slacks(H_raw, problem)
    G_is_free = _bool_metadata(problem, 'G_is_free', len(G_shifted))
    residuals = np.minimum(np.abs(G_shifted), np.abs(H_shifted)).copy()
    for i in range(len(residuals)):
        if G_is_free[i]:
            residuals[i] = abs(float(H_shifted[i]))

    if not ubH_slacks:
        return float(np.max(residuals))

    for i, upper_slack in ubH_slacks.items():
        if G_is_free[i]:
            lower = abs(float(H_shifted[i]))
            upper = abs(float(upper_slack))
        else:
            lower = min(abs(float(G_shifted[i])), abs(float(H_shifted[i])))
            upper = min(abs(float(G_shifted[i])), abs(float(upper_slack)))
        residuals[i] = min(lower, upper)
    return float(np.max(residuals))


def benchmark_feas_res(x: np.ndarray, problem: Dict[str, Any]) -> float:
    # Suite-appropriate complementarity residual for benchmark reporting.
    return mcp_feasibility_residual(x, problem)


def biactive_indices(x: np.ndarray, problem: Dict[str, Any], tol: float = 1e-6) -> List[int]:
    # Return indices where both |G_shifted_i| < tol and |H_shifted_i| < tol.
    G_shifted, H_shifted = _get_shifted_GH(x, problem)
    if len(G_shifted) == 0:
        return []

    G_raw, H_raw = _evaluate_GH_raw(x, problem)
    ubH_map = {int(i): float(ub) for i, ub in problem.get('ubH_finite', [])}

    mask = np.zeros(len(G_shifted), dtype=bool)
    for i in range(len(G_shifted)):
        g_active = abs(float(G_shifted[i])) < tol
        h_lower_active = abs(float(H_shifted[i])) < tol
        h_upper_active = i in ubH_map and abs(float(ubH_map[i]) - float(H_raw[i])) < tol
        mask[i] = g_active and (h_lower_active or h_upper_active)
    return list(np.where(mask)[0])


def complementarity_residual(x: np.ndarray, problem: Dict[str, Any]) -> float:
    # Backward-compatible alias; use homotopy_comp_res() in new code.
    return homotopy_comp_res(x, problem)
