# Convert MPECLib GAMS (.gms) files to CasADi JSON (.nl.json)

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import casadi as ca
import numpy as np

logger = logging.getLogger("mpecss.convert_mpeclib")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

_BIG = 1e20

# GAMS helpers

_GAMS_REPLACEMENTS = [
    # Map GAMS functions to CasADi names.
    (re.compile(r'\bPOWER\s*\(', re.IGNORECASE), 'pow_('),
    (re.compile(r'\bSQR\s*\(',   re.IGNORECASE), 'sqr_('),
    (re.compile(r'\bSQRT\s*\(',  re.IGNORECASE), 'sqrt_('),
    (re.compile(r'\bLOG\s*\(',   re.IGNORECASE), 'log_('),
    (re.compile(r'\bEXP\s*\(',   re.IGNORECASE), 'exp_('),
    (re.compile(r'\bABS\s*\(',   re.IGNORECASE), 'abs_('),
    (re.compile(r'\bSIN\s*\(',   re.IGNORECASE), 'sin_('),
    (re.compile(r'\bCOS\s*\(',   re.IGNORECASE), 'cos_('),
    (re.compile(r'\bTAN\s*\(',   re.IGNORECASE), 'tan_('),
    (re.compile(r'\bSIGN\s*\(',  re.IGNORECASE), 'sign_('),
    (re.compile(r'\bMIN\s*\(',   re.IGNORECASE), 'fmin_('),
    (re.compile(r'\bMAX\s*\(',   re.IGNORECASE), 'fmax_('),
]

_CASADI_NS: Dict[str, Any] = {
    "pow_":  lambda x, y: x ** y,
    "sqr_":  lambda x:    x ** 2,
    "sqrt_": ca.sqrt,
    "log_":  ca.log,
    "exp_":  ca.exp,
    "abs_":  ca.fabs,
    "sin_":  ca.sin,
    "cos_":  ca.cos,
    "tan_":  ca.tan,
    "sign_": ca.sign,
    "fmin_": ca.fmin,
    "fmax_": ca.fmax,
}


# Translate a GAMS expression to Python/CasADi form.
def _gams_to_python(expr: str) -> str:
    for pattern, replacement in _GAMS_REPLACEMENTS:
        expr = pattern.sub(replacement, expr)
    return expr


# Evaluate a GAMS expression in a CasADi namespace.
def _eval_expr(expr_str: str, sym_map: Dict[str, ca.SX]) -> ca.SX:
    expr_str = _gams_to_python(expr_str.strip())
    ns = dict(_CASADI_NS)
    ns.update(sym_map)
    try:
        return eval(expr_str, {"__builtins__": {}}, ns)  # noqa: S307
    except Exception as exc:
        raise ValueError(f"Failed to eval expression: {expr_str!r}  ({exc})") from exc


# GMS parser

class GmsParser:
    def __init__(self, text: str) -> None:
        self.variables: List[str] = []
        self.positive:  set       = set()
        self.free:      set       = set()
        self.equations: List[str] = []
        self.eq_def:    Dict[str, Tuple[str, str, str]] = {}
        self.lb:        Dict[str, float] = {}
        self.ub:        Dict[str, float] = {}
        self.l0:        Dict[str, float] = {}
        self.fx:        Dict[str, float] = {}
        self.model_pairs: List[Tuple[str, Optional[str]]] = []
        self.obj_var:   str = "objvar"
        self._parse(text)

    # Parsing helpers

    @staticmethod
    def _clean(text: str) -> str:
        lines = []
        for line in text.splitlines():
            if line.lstrip().startswith('*'):
                lines.append('')
            else:
                lines.append(line)
        return '\n'.join(lines)

    @staticmethod
    def _strip_semicolon(s: str) -> str:
        s = s.strip()
        if s.endswith(';'):
            s = s[:-1].strip()
        return s

    # Split a name list.
    @staticmethod
    def _split_names(s: str) -> List[str]:
        return [n.strip() for n in re.split(r'[,\s]+', s) if n.strip()]

    # Parse the file.
    def _parse(self, raw: str) -> None:
        text = self._clean(raw)
        one_line = ' '.join(text.split())

        self._parse_variables(one_line)
        self._parse_equations_decl(one_line)
        self._parse_equation_defs(one_line)
        self._parse_bounds(one_line)
        self._parse_model(one_line)
        self._parse_solve(one_line)

    def _parse_variables(self, text: str) -> None:
        m = re.search(r'\bVariables\s+(.*?);', text, re.IGNORECASE)
        if m:
            self.variables = self._split_names(m.group(1))

        for pm in re.finditer(r'\bPositive\s+Variables\s+(.*?);', text, re.IGNORECASE):
            self.positive.update(self._split_names(pm.group(1)))

        for pm in re.finditer(r'\bFree\s+Variables\s+(.*?);', text, re.IGNORECASE):
            self.free.update(self._split_names(pm.group(1)))

    def _parse_equations_decl(self, text: str) -> None:
        m = re.search(r'\bEquations\s+(.*?);', text, re.IGNORECASE)
        if m:
            self.equations = self._split_names(m.group(1))

    # Find equation definitions.
    def _parse_equation_defs(self, text: str) -> None:
        pattern = re.compile(
            r'\b(e\d+)\s*\.\.\s*(.*?)\s*=(E|G|L)=\s*(.*?)\s*;',
            re.IGNORECASE
        )
        for m in pattern.finditer(text):
            name = m.group(1).lower()
            lhs  = m.group(2).strip()
            typ  = m.group(3).upper()
            rhs  = m.group(4).strip()
            self.eq_def[name] = (lhs, typ, rhs)

    # Parse bounds and initial values.
    def _parse_bounds(self, text: str) -> None:
        lo_pat = re.compile(r'\b(\w+)\.lo\s*=\s*([^;]+?)\s*;', re.IGNORECASE)
        up_pat = re.compile(r'\b(\w+)\.up\s*=\s*([^;]+?)\s*;', re.IGNORECASE)
        fx_pat = re.compile(r'\b(\w+)\.fx\s*=\s*([^;]+?)\s*;', re.IGNORECASE)
        lv_pat = re.compile(r'\b(\w+)\.l\s*=\s*([^;]+?)\s*;',  re.IGNORECASE)

        for m in lo_pat.finditer(text):
            self.lb[m.group(1)] = float(m.group(2))
        for m in up_pat.finditer(text):
            self.ub[m.group(1)] = float(m.group(2))
        for m in fx_pat.finditer(text):
            val = float(m.group(2))
            self.fx[m.group(1)] = val
            self.lb[m.group(1)] = val
            self.ub[m.group(1)] = val
        for m in lv_pat.finditer(text):
            self.l0[m.group(1)] = float(m.group(2))

    # Parse the model statement.
    def _parse_model(self, text: str) -> None:
        m = re.search(r'\bModel\s+\w+\s*/\s*(.*?)\s*/\s*;', text, re.IGNORECASE)
        if not m:
            raise ValueError("No Model statement found in GMS file")
        body = m.group(1)
        for token in re.split(r'\s*,\s*', body):
            token = token.strip()
            if not token:
                continue
            if '.' in token:
                parts = token.split('.')
                eq_name  = parts[0].strip().lower()
                var_name = parts[1].strip()
                self.model_pairs.append((eq_name, var_name))
            else:
                self.model_pairs.append((token.lower(), None))

    # Parse the solve statement.
    def _parse_solve(self, text: str) -> None:
        m = re.search(r'\bSolve\s+\w+\s+using\s+MPEC\s+minimizing\s+(\w+)\s*;',
                      text, re.IGNORECASE)
        if m:
            self.obj_var = m.group(1)


# Problem builder

# Build the CasADi problem data for one file.
def _build_problem(parser: GmsParser, name: str) -> Dict[str, Any]:
    obj_var = parser.obj_var

    dec_vars = [v for v in parser.variables if v != obj_var]
    n_x = len(dec_vars)
    if n_x == 0:
        raise ValueError(f"{name}: no decision variables found")

    lbx: List[float] = []
    ubx: List[float] = []
    w0:  List[float] = []

    for v in dec_vars:
        # Positive variables start at 0 unless overridden.
        lb_default = 0.0 if (v in parser.positive and v not in parser.free) else -_BIG
        lb = parser.lb.get(v, lb_default)
        ub = parser.ub.get(v, _BIG)
        if v in parser.fx:
            lb = ub = parser.fx[v]
        # Use the level, then the lower bound, then 0.
        l0_default = lb if lb > -_BIG else 0.0
        l0 = parser.l0.get(v, l0_default)
        lbx.append(lb)
        ubx.append(ub)
        w0.append(l0)

    sym_cls = ca.SX if n_x <= 500 else ca.MX
    w_sym   = sym_cls.sym("w", n_x)
    sym_map: Dict[str, Any] = {v: w_sym[i] for i, v in enumerate(dec_vars)}

    # Keep objvar until the objective is found.
    sym_map[obj_var] = sym_cls.sym(obj_var)  # placeholder, replaced below

    # Returns (lhs_expr, type, rhs_expr) for equation eq_name.
    def _parse_eq(eq_name: str) -> Tuple[ca.SX, str, ca.SX]:
        if eq_name not in parser.eq_def:
            raise ValueError(f"{name}: equation {eq_name!r} has no definition")
        lhs_str, typ, rhs_str = parser.eq_def[eq_name]
        lhs = _eval_expr(lhs_str, sym_map)
        rhs = _eval_expr(rhs_str, sym_map)
        return lhs, typ, rhs

    def _extract_objective() -> ca.SX:
        for eq_name, paired_var in parser.model_pairs:
            if paired_var is not None:
                continue
            lhs_str, typ, rhs_str = parser.eq_def.get(eq_name, ("", "", ""))
            if typ != 'E':
                continue
            full_str = f"({lhs_str}) - ({rhs_str})"
            if re.search(r'\b' + re.escape(obj_var) + r'\b', full_str, re.IGNORECASE):
                # Assume objvar is linear here.
                ov_sym = sym_map[obj_var]
                full_expr = _eval_expr(full_str, sym_map)
                coef = float(ca.jacobian(full_expr, ov_sym))
                if abs(coef) < 1e-12:
                    raise ValueError(f"{name}: objvar coefficient is zero in {eq_name}")
                sym_map_no_ov = {**sym_map, obj_var: ca.SX(0)}
                intercept = _eval_expr(full_str, sym_map_no_ov)
                f_expr = -intercept / coef
                return ca.simplify(f_expr)
        raise ValueError(f"{name}: could not locate objective equation for {obj_var!r}")

    f_expr = _extract_objective()
    # objvar is not needed anymore.
    sym_map.pop(obj_var, None)

    obj_eq_names: set = set()
    for eq_name, paired_var in parser.model_pairs:
        if paired_var is not None:
            continue
        lhs_str, typ, _ = parser.eq_def.get(eq_name, ("", "", ""))
        if re.search(r'\b' + re.escape(obj_var) + r'\b', lhs_str, re.IGNORECASE):
            obj_eq_names.add(eq_name)

    comp_pairs: List[Tuple[str, str]] = []
    reg_eq_names: List[str] = []

    for eq_name, paired_var in parser.model_pairs:
        if eq_name in obj_eq_names:
            continue
        if paired_var is None:
            reg_eq_names.append(eq_name)
        else:
            _, typ, _ = parser.eq_def.get(eq_name, ("", "E", ""))
            if typ in ('G', 'L'):
                comp_pairs.append((eq_name, paired_var))
            else:
                reg_eq_names.append(eq_name)

    # Build complementarity terms.
    G_parts: List[ca.SX] = []
    H_parts: List[ca.SX] = []
    lbG: List[float] = []
    lbH: List[float] = []

    for eq_name, var_name in comp_pairs:
        lhs, typ, rhs = _parse_eq(eq_name)
        if typ == 'G':
            g_expr = lhs - rhs
        else:
            g_expr = rhs - lhs

        if var_name not in sym_map:
            raise ValueError(f"{name}: comp variable {var_name!r} not in sym_map")
        var_sym = sym_map[var_name]
        var_idx = dec_vars.index(var_name)
        h_lb = lbx[var_idx]
        h_expr = var_sym - ca.DM(h_lb) if h_lb != 0.0 else var_sym

        G_parts.append(ca.reshape(g_expr, -1, 1))
        H_parts.append(ca.reshape(h_expr, -1, 1))
        lbG.append(0.0)
        lbH.append(0.0)

    if not G_parts:
        raise ValueError(f"{name}: no complementarity pairs found")

    G_expr = ca.vertcat(*G_parts)
    H_expr = ca.vertcat(*H_parts)

    # Build regular constraints.
    g_parts: List[ca.SX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    for eq_name in reg_eq_names:
        if eq_name in obj_eq_names:
            continue
        lhs, typ, rhs = _parse_eq(eq_name)
        residual = lhs - rhs
        if typ == 'E':
            g_parts.append(ca.reshape(residual, -1, 1))
            n_rows = int(ca.size1(residual))
            lbg.extend([0.0]  * n_rows)
            ubg.extend([0.0]  * n_rows)
        elif typ == 'G':
            g_parts.append(ca.reshape(residual, -1, 1))
            n_rows = int(ca.size1(residual))
            lbg.extend([0.0]  * n_rows)
            ubg.extend([_BIG] * n_rows)
        else:
            g_parts.append(ca.reshape(residual, -1, 1))
            n_rows = int(ca.size1(residual))
            lbg.extend([-_BIG] * n_rows)
            ubg.extend([0.0]  * n_rows)

    # Serialise CasADi functions.
    f_fn = ca.Function("f_fun", [w_sym], [f_expr])
    G_fn = ca.Function("G_fun", [w_sym], [G_expr])
    H_fn = ca.Function("H_fun", [w_sym], [H_expr])

    problem: Dict[str, Any] = {
        "name":    name,
        "lbw":     lbx,
        "ubw":     ubx,
        "w0":      w0,
        "f_fun":   f_fn.serialize(),
        "G_fun":   G_fn.serialize(),
        "H_fun":   H_fn.serialize(),
        "lbG":     lbG,
        "ubG":     [_BIG] * len(lbG),
        "lbH":     lbH,
        "ubH":     [_BIG] * len(lbH),
    }

    if g_parts:
        g_expr_full = ca.vertcat(*g_parts)
        g_fn = ca.Function("g_fun", [w_sym], [g_expr_full])
        problem["g_fun"] = g_fn.serialize()
        problem["lbg"]   = lbg
        problem["ubg"]   = ubg

    return problem


# File conversion

# Convert one .gms file to .nl.json.
def convert_gms_to_json(gms_path: str, out_path: str, dry_run: bool = False) -> bool:
    name = os.path.basename(gms_path).replace(".gms", "")
    try:
        with open(gms_path, "r", encoding="utf-8") as f:
            text = f.read()

        parser = GmsParser(text)
        problem = _build_problem(parser, name)
        n_comp = len(problem["lbG"])

        if dry_run:
            n_x = len(problem["lbw"])
            n_g = len(problem.get("lbg", []))
            logger.info(f"[DRY] {name:30s}  n_x={n_x:5d}  n_comp={n_comp:4d}  n_con={n_g:4d}")
            return True

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(problem, f, separators=(",", ":"))

        n_x = len(problem["lbw"])
        n_g = len(problem.get("lbg", []))
        logger.info(f"  ✓  {name:30s}  n_x={n_x:5d}  n_comp={n_comp:4d}  n_con={n_g:4d}  → {out_path}")
        return True

    except Exception as exc:
        logger.error(f"  ✗  {name}: {exc}")
        return False


# Command line entry point

def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    parser = argparse.ArgumentParser(
        description="Convert MPECLib GAMS (.gms) files to CasADi JSON (.nl.json)"
    )
    parser.add_argument(
        "--gms-dir", default=os.path.join(root, "benchmarks", "mpeclib", "mpeclib-gms"),
        help="Directory containing .gms input files"
    )
    parser.add_argument(
        "--out", default=os.path.join(root, "benchmarks", "mpeclib", "mpeclib-json"),
        help="Output directory for .nl.json files"
    )
    parser.add_argument(
        "--file", default=None,
        help="Convert a single .gms file (overrides --gms-dir)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse only — do not write any JSON files"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.file:
        gms_files = [os.path.abspath(args.file)]
    else:
        if not os.path.isdir(args.gms_dir):
            logger.error(f"GMS directory not found: {args.gms_dir}")
            sys.exit(1)
        gms_files = sorted(glob.glob(os.path.join(args.gms_dir, "*.gms")))

    if not gms_files:
        logger.error("No .gms files found")
        sys.exit(1)

    logger.info(f"Converting {len(gms_files)} GMS file(s)  →  {args.out}")
    n_ok = n_fail = 0

    for gms_path in gms_files:
        base = os.path.basename(gms_path).replace(".gms", "")
        out_path = os.path.join(args.out, base + ".nl.json")
        ok = convert_gms_to_json(gms_path, out_path, dry_run=args.dry_run)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    logger.info(f"\nDone: {n_ok} succeeded, {n_fail} failed out of {len(gms_files)} total")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
