"""MacMPEC JSON loader for solver-ready problem dictionaries."""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List

import casadi as ca
import numpy as np

_BIG = 1e20
_X0_PERTURBATION = 0.01


def _sanitize_bound(value: float | None, default: float) -> float:
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


def _sanitize_bounds(values: List[float] | None, default: float) -> List[float]:
    if values is None:
        return []
    if isinstance(values, (int, float)):
        values = [values]
    return [_sanitize_bound(v, default) for v in values]


def evaluate_GH(x: np.ndarray, problem: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    G = np.asarray(problem["G_fn"](x)).flatten()
    H = np.asarray(problem["H_fn"](x)).flatten()
    return G, H


def complementarity_residual(x: np.ndarray, problem: Dict[str, Any]) -> float:
    # Compute the canonical shifted, sign-aware MCP residual.
    from mpecss.helpers.comp_residuals import complementarity_residual as _canonical_residual

    return _canonical_residual(x, problem)


def biactive_indices(x: np.ndarray, problem: Dict[str, Any], sta_tol: float = 1e-6) -> List[int]:
    # Return canonical shifted biactive indices.
    from mpecss.helpers.comp_residuals import biactive_indices as _canonical_biactive

    return _canonical_biactive(x, problem, tol=sta_tol)


def load_macmpec(filepath: str) -> Dict[str, Any]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"MacMPEC file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    name = os.path.basename(filepath).replace(".nl.json", "")
    lbx = _sanitize_bounds(data.get("lbw", []), -_BIG)
    ubx = _sanitize_bounds(data.get("ubw", []), _BIG)
    w0 = np.array(data.get("w0", []), dtype=float)
    n_x = len(lbx)

    f_fn = ca.Function.deserialize(data["f_fun"])
    G_fn = ca.Function.deserialize(data["G_fun"])
    H_fn = ca.Function.deserialize(data["H_fun"])
    g_fn = ca.Function.deserialize(data["g_fun"]) if data.get("g_fun") else None
    lbg = _sanitize_bounds(data.get("lbg", []), -_BIG)
    ubg = _sanitize_bounds(data.get("ubg", []), _BIG)

    lbG_raw = data.get("lbG", [0.0])
    if isinstance(lbG_raw, (int, float)):
        lbG_raw = [lbG_raw]
    G_is_free = [
        v is None or (isinstance(v, float) and not np.isfinite(v) and v < 0)
        for v in lbG_raw
    ]
    lbG = _sanitize_bounds(lbG_raw, 0.0)   # -inf → 0.0 (shift base for bounded side)
    lbH = _sanitize_bounds(data.get("lbH", [0.0]), 0.0)
    n_comp = len(lbG)
    n_con = len(lbg)

    def _finite_ubs(raw, n):
        # Return list of (index, value) for finite upper bounds.
        if raw is None:
            return []
        if isinstance(raw, (int, float)):
            raw = [raw] * n
        result = []
        for i, v in enumerate(raw[:n]):
            sv = _sanitize_bound(v, _BIG)
            if sv < 1e19:          # finite upper bound
                result.append((i, sv))
        return result

    ubH_finite = _finite_ubs(data.get("ubH"), n_comp)   # [(idx, val), ...]
    ubG_finite = _finite_ubs(data.get("ubG"), n_comp)

    def x0_fn(seed: int = 0) -> np.ndarray:
        rng = np.random.RandomState(seed)
        x0 = w0.copy()
        x0 += rng.uniform(-_X0_PERTURBATION, _X0_PERTURBATION, size=x0.shape)
        return np.clip(x0, np.array(lbx) + 1e-8, np.array(ubx) - 1e-8)

    def build_casadi(t_k: float, delta_k: float, smoothing: str = "product") -> Dict[str, Any]:
        x = ca.MX.sym("x", n_x) if n_x > 500 else ca.SX.sym("x", n_x)
        f = f_fn(x)

        G_raw = G_fn(x)
        H = H_fn(x) - ca.DM(lbH)

        G = ca.vcat([
            G_raw[i]          if G_is_free[i] else G_raw[i] - lbG[i]
            for i in range(n_comp)
        ])

        g_parts = []
        lbg_parts: List[float] = []
        ubg_parts: List[float] = []

        if g_fn is not None:
            g_parts.append(g_fn(x))
            lbg_parts.extend(lbg)
            ubg_parts.extend(ubg)

        ubH_idx = {i for i, _ in ubH_finite}
        bounded_idx = [
            i for i in range(n_comp)
            if (not G_is_free[i]) or (i not in ubH_idx)
        ]
        if bounded_idx:
            g_parts.append(ca.vcat([G[i] + delta_k for i in bounded_idx]))
            lbg_parts.extend([0.0] * len(bounded_idx))
            ubg_parts.extend([_BIG] * len(bounded_idx))

        g_parts.append(H + delta_k)
        lbg_parts.extend([0.0] * n_comp)
        ubg_parts.extend([_BIG] * n_comp)

        if ubH_finite:
            g_parts.append(ca.vcat([
                ca.DM(ub - lbH[i]) - H[i] + delta_k for i, ub in ubH_finite
            ]))
            lbg_parts.extend([0.0] * len(ubH_finite))
            ubg_parts.extend([_BIG] * len(ubH_finite))

            g_parts.append(ca.vcat([
                (-G[i]) * (ca.DM(ub - lbH[i]) - H[i]) - t_k for i, ub in ubH_finite
            ]))
            lbg_parts.extend([-_BIG] * len(ubH_finite))
            ubg_parts.extend([0.0]  * len(ubH_finite))

        if ubG_finite:
            bounded_ub_G_idx = [i for i, _ in ubG_finite if not G_is_free[i]]
            if bounded_ub_G_idx:
                g_parts.append(ca.vcat([
                    ca.DM(ub) - G[i] + delta_k for i, ub in ubG_finite
                    if not G_is_free[i]
                ]))
                lbg_parts.extend([0.0] * len(bounded_ub_G_idx))
                ubg_parts.extend([_BIG] * len(bounded_ub_G_idx))

        if smoothing == "fb":
            comp = ca.sqrt(G**2 + H**2) - G - H - t_k
            g_parts.append(comp)
            lbg_parts.extend([-_BIG] * n_comp)
            ubg_parts.extend([0.0] * n_comp)
        else:
            g_parts.append(ca.vcat([G[i] * H[i] - t_k for i in range(n_comp)]))
            lbg_parts.extend([-_BIG] * n_comp)
            ubg_parts.extend([0.0] * n_comp)

        n_bounded_g = len(bounded_idx)
        n_ubH_blocks = len(ubH_finite)
        off_G_lb    = n_con
        off_H_lb    = off_G_lb   + n_bounded_g
        off_ubH_lb  = off_H_lb   + n_comp
        off_ubH_uc  = off_ubH_lb + n_ubH_blocks   # upper-comp block
        off_comp    = off_ubH_uc + n_ubH_blocks

        return {
            "x": x,
            "f": f,
            "g": ca.vertcat(*g_parts),
            "lbg": lbg_parts,
            "ubg": ubg_parts,
            "lbx": lbx,
            "ubx": ubx,
            "n_comp": n_comp,
            "n_orig_con": n_con,
            "n_bounded_G":    n_bounded_g,
            "n_ubH":          n_ubH_blocks,
            "off_G_lb":       off_G_lb,
            "off_H_lb":       off_H_lb,
            "off_comp":       off_comp,
            "_bounded_G_idx": bounded_idx,   # which comp indices have G>=0 blocks
        }

    return {
        "name": name,
        "family": "macmpec",
        "n_x": n_x,
        "n_comp": n_comp,
        "n_con": n_con,
        "n_p": 0,
        "x0_fn": x0_fn,
        "build_casadi": build_casadi,
        "f_fn": f_fn,
        "G_fn": G_fn,
        "H_fn": H_fn,
        "lbx": lbx,
        "ubx": ubx,
        "G_is_free": G_is_free,
        "lbG_eff": lbG,
        "lbH_eff": lbH,
        "ubH_finite": ubH_finite,
        "ubG_finite": ubG_finite,
        "_source_path": filepath,
    }


def load_macmpec_batch(directory: str, pattern: str = "*.nl.json") -> List[Dict[str, Any]]:
    return [load_macmpec(fp) for fp in sorted(glob.glob(os.path.join(directory, pattern)))]


def get_problem(name: str, macmpec_dir: str | None = None) -> Dict[str, Any]:
    if os.path.isfile(name):
        return load_macmpec(name)
    if macmpec_dir is None:
        raise FileNotFoundError(f"Could not resolve problem path: {name}")
    return load_macmpec(os.path.join(macmpec_dir, f"{name}.nl.json"))
