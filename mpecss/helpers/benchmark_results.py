# Result summary extraction and column mapping for benchmark output.

import copy
from typing import Dict, Any, Optional, Callable

import numpy as np

from mpecss.helpers.benchmark_audit import _point_fingerprint, _json_safe


def _summarize_result_state(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not result:
        return None

    summary: Dict[str, Any] = {
        "status": result.get("status"),
        "stationarity": result.get("stationarity"),
        "f_final": result.get("f_final"),
        "comp_res": result.get("comp_res"),
        "kkt_res": result.get("kkt_res"),
        "sign_test_pass": result.get("sign_test_pass"),
        "b_stationarity": result.get("b_stationarity"),
        "lpec_obj": result.get("lpec_obj"),
        "licq_holds": result.get("licq_holds"),
        "n_outer_iters": result.get("n_outer_iters"),
        "n_restorations": result.get("n_restorations"),
        "cpu_time": result.get("cpu_time"),
        "log_count": len(result.get("logs", []) or []),
    }
    summary.update(_point_fingerprint(result.get("z_final")))

    phase_i = result.get("phase_i_result") or {}
    if phase_i:
        summary["phase_i"] = {
            "success": phase_i.get("success"),
            "cpu_time": phase_i.get("cpu_time"),
            "n_attempts": phase_i.get("n_attempts"),
            "initial_comp_res": phase_i.get("initial_comp_res"),
            "final_comp_res": phase_i.get("final_comp_res"),
            "feasibility_achieved": phase_i.get("feasibility_achieved"),
            "near_feasibility": phase_i.get("near_feasibility"),
        }

    bnlp = result.get("bnlp_polish") or {}
    if bnlp:
        summary["bnlp_polish"] = {
            "accepted": bnlp.get("accepted"),
            "status": bnlp.get("status"),
            "success": bnlp.get("success"),
            "f_val": bnlp.get("f_val"),
            "original_f_val": bnlp.get("original_f_val"),
            "comp_res_polish": bnlp.get("comp_res_polish"),
            "improvement": bnlp.get("improvement"),
        }

    lpec = result.get("lpec_refine") or {}
    if lpec:
        summary["lpec_refine"] = {
            "bstat_found": lpec.get("bstat_found"),
            "n_outer": lpec.get("n_outer"),
            "n_inner_total": lpec.get("n_inner_total"),
            "n_bnlps": lpec.get("n_bnlps"),
            "n_lpecs": lpec.get("n_lpecs"),
            "improvement": lpec.get("improvement"),
            "cpu_time": lpec.get("cpu_time"),
        }

    bstat = result.get("bstat_details") or {}
    if bstat:
        summary["bstat_details"] = {
            "lpec_status": bstat.get("lpec_status"),
            "classification": bstat.get("classification"),
            "licq_holds": bstat.get("licq_holds"),
            "licq_rank": bstat.get("licq_rank"),
            "lpec_obj": bstat.get("lpec_obj"),
            "timed_out": bstat.get("timed_out"),
            "elapsed_s": bstat.get("elapsed_s"),
        }

    return _json_safe(summary)


def _max_box_violation(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    lower_mask = np.isfinite(lower) & (lower > -1e19)
    upper_mask = np.isfinite(upper) & (upper < 1e19)
    lower_violation = np.maximum(lower - values, 0.0) if np.any(lower_mask) else np.zeros_like(values)
    upper_violation = np.maximum(values - upper, 0.0) if np.any(upper_mask) else np.zeros_like(values)
    if np.any(~lower_mask):
        lower_violation = np.where(lower_mask, lower_violation, 0.0)
    if np.any(~upper_mask):
        upper_violation = np.where(upper_mask, upper_violation, 0.0)
    return float(max(np.max(lower_violation, initial=0.0), np.max(upper_violation, initial=0.0)))


def _build_point_diagnostic_evaluator(problem: Dict[str, Any]) -> Callable[[Any], Dict[str, Any]]:
    import casadi as ca
    from mpecss.helpers.comp_residuals import benchmark_feas_res, biactive_residual

    info = problem["build_casadi"](0.0, 0.0, smoothing="product")
    x_sym = info["x"]
    f_eval = ca.Function("audit_f_eval", [x_sym], [info["f"]])
    g_eval = ca.Function("audit_g_eval", [x_sym], [info["g"]])

    lbg = np.asarray(info.get("lbg", []), dtype=float).flatten()
    ubg = np.asarray(info.get("ubg", []), dtype=float).flatten()
    lbx = np.asarray(info.get("lbx", problem.get("lbx", [])), dtype=float).flatten()
    ubx = np.asarray(info.get("ubx", problem.get("ubx", [])), dtype=float).flatten()
    n_orig_con = int(info.get("n_orig_con", problem.get("n_con", 0)) or 0)

    def evaluate(z: Any) -> Dict[str, Any]:
        arr = np.asarray(z, dtype=float).flatten()
        diagnostics: Dict[str, Any] = {}
        diagnostics.update(_point_fingerprint(arr))
        diagnostics["all_finite"] = bool(arr.size == 0 or np.all(np.isfinite(arr)))

        objective_eval = None
        g_val = np.array([])
        try:
            objective_eval = float(f_eval(arr))
        except Exception:
            diagnostics["all_finite"] = False

        try:
            g_val = np.asarray(g_eval(arr)).flatten()
        except Exception:
            diagnostics["all_finite"] = False

        orig_violation = 0.0
        if g_val.size and n_orig_con > 0:
            orig_g = g_val[:n_orig_con]
            orig_lbg = lbg[:n_orig_con] if lbg.size else np.full(n_orig_con, -np.inf)
            orig_ubg = ubg[:n_orig_con] if ubg.size else np.full(n_orig_con, np.inf)
            orig_violation = _max_box_violation(orig_g, orig_lbg, orig_ubg)

        var_violation = _max_box_violation(arr, lbx, ubx) if lbx.size and ubx.size else 0.0

        comp_side_violation = 0.0
        try:
            G = np.asarray(problem["G_fn"](arr)).flatten()
            H = np.asarray(problem["H_fn"](arr)).flatten()
            diagnostics["all_finite"] = diagnostics["all_finite"] and bool(
                np.all(np.isfinite(G)) and np.all(np.isfinite(H))
            )

            lbG_eff = np.asarray(problem.get("lbG_eff", np.zeros(len(G))), dtype=float)
            lbH_eff = np.asarray(problem.get("lbH_eff", np.zeros(len(H))), dtype=float)
            G_is_free = list(problem.get("G_is_free", [False] * len(G)))
            lower_G = [
                max(float(lbG_eff[i] - G[i]), 0.0)
                for i in range(len(G))
                if i < len(G_is_free) and not G_is_free[i]
            ]
            lower_H = [max(float(lbH_eff[i] - H[i]), 0.0) for i in range(len(H))]
            upper_G = [max(float(G[i] - ub), 0.0) for i, ub in problem.get("ubG_finite", [])]
            upper_H = [max(float(H[i] - ub), 0.0) for i, ub in problem.get("ubH_finite", [])]
            comp_side_violation = float(max(lower_G + lower_H + upper_G + upper_H + [0.0]))
        except Exception:
            diagnostics["all_finite"] = False

        try:
            bench_comp = benchmark_feas_res(arr, problem)
        except Exception:
            bench_comp = None
            diagnostics["all_finite"] = False
        try:
            biactive_res = biactive_residual(arr, problem)
        except Exception:
            biactive_res = None
            diagnostics["all_finite"] = False

        diagnostics.update(
            {
                "objective_eval": objective_eval,
                "benchmark_feas_res": bench_comp,
                "biactive_res": biactive_res,
                "orig_constr_violation": orig_violation,
                "var_bound_violation": var_violation,
                "comp_side_violation": comp_side_violation,
                "overall_primal_violation": max(orig_violation, var_violation, comp_side_violation),
            }
        )
        return _json_safe(diagnostics)

    return evaluate


def _apply_raw_summary_columns(row: Dict[str, Any], raw_summary: Optional[Dict[str, Any]]) -> None:
    if not raw_summary:
        return

    mappings = {
        "raw_status": "status",
        "raw_stationarity": "stationarity",
        "raw_f_final": "f_final",
        "raw_comp_res": "comp_res",
        "raw_kkt_res": "kkt_res",
        "raw_sign_test_pass": "sign_test_pass",
        "raw_b_stationarity": "b_stationarity",
        "raw_lpec_obj": "lpec_obj",
        "raw_licq_holds": "licq_holds",
        "raw_n_outer_iters": "n_outer_iters",
        "raw_n_restorations": "n_restorations",
        "raw_cpu_time_total": "cpu_time",
        "raw_point_sha256": "point_sha256",
    }
    for column, key in mappings.items():
        row[column] = raw_summary.get(key)

    bstat = raw_summary.get("bstat_details") or {}
    row["raw_bstat_lpec_status"] = bstat.get("lpec_status")
    row["raw_bstat_classification"] = bstat.get("classification")

    phase_i = raw_summary.get("phase_i") or {}
    row["raw_time_phase_i"] = phase_i.get("cpu_time")


def _apply_point_diagnostic_columns(
    row: Dict[str, Any],
    prefix: str,
    diagnostics: Optional[Dict[str, Any]],
    reported_objective: Optional[float],
) -> None:
    if not diagnostics:
        return

    row[f"{prefix}_benchmark_feas_res"] = diagnostics.get("benchmark_feas_res")
    row[f"{prefix}_biactive_res"] = diagnostics.get("biactive_res")
    row[f"{prefix}_orig_constr_violation"] = diagnostics.get("orig_constr_violation")
    row[f"{prefix}_var_bound_violation"] = diagnostics.get("var_bound_violation")
    row[f"{prefix}_comp_side_violation"] = diagnostics.get("comp_side_violation")
    row[f"{prefix}_overall_primal_violation"] = diagnostics.get("overall_primal_violation")
    row[f"{prefix}_objective_eval"] = diagnostics.get("objective_eval")
    row[f"{prefix}_point_sha256"] = diagnostics.get("point_sha256")

    objective_eval = diagnostics.get("objective_eval")
    if objective_eval is not None and reported_objective is not None:
        try:
            row[f"{prefix}_objective_abs_diff"] = abs(float(objective_eval) - float(reported_objective))
        except Exception:
            row[f"{prefix}_objective_abs_diff"] = None


def map_iteration_to_snapshot(log, prefix: str) -> Dict[str, Any]:
    return {
        prefix + "t_k": log.t_k,
        prefix + "delta_k": log.delta_k,
        prefix + "comp_res": log.comp_res,
        prefix + "kkt_res": log.kkt_res,
        prefix + "objective": log.objective,
        prefix + "sign_test": log.sign_test,
        prefix + "solver_status": log.solver_status,
        prefix + "n_biactive": log.n_biactive,
        prefix + "nlp_iters": log.nlp_iter_count,
        prefix + "solver_type": log.solver_type,
        prefix + "warmstart": log.warmstart_type,
        prefix + "t_update_regime": log.t_update_regime,
        prefix + "cpu_time": log.cpu_time,
        prefix + "sta_tol": log.sta_tol,
        prefix + "improvement_ratio": log.improvement_ratio,
        prefix + "stagnation_count": log.stagnation_count,
        prefix + "tracking_count": log.tracking_count,
        prefix + "is_tracking": log.is_in_tracking_regime,
        prefix + "solver_fallback": log.solver_fallback_occurred,
        prefix + "consec_fails": log.consecutive_solver_failures,
        prefix + "best_comp_so_far": log.best_comp_res_so_far,
        prefix + "best_iter_achieved": log.best_iter_achieved,
        prefix + "ipopt_tol_used": log.ipopt_tol_used,
        prefix + "restoration_used": log.restoration_used,
        prefix + "restoration_trigger": log.restoration_trigger_reason,
        prefix + "restoration_success": log.restoration_success,
        prefix + "biactive_indices": log.biactive_indices_str,
        prefix + "lambda_G_min": log.lambda_G_min,
        prefix + "lambda_G_max": log.lambda_G_max,
        prefix + "lambda_H_min": log.lambda_H_min,
        prefix + "lambda_H_max": log.lambda_H_max,
    }


def _infer_final_result_source(
    raw_summary: Optional[Dict[str, Any]],
    bnlp_summary: Optional[Dict[str, Any]],
    final_summary: Optional[Dict[str, Any]],
) -> Optional[str]:
    if not raw_summary or not final_summary:
        return None
    raw_hash = raw_summary.get("point_sha256")
    final_hash = final_summary.get("point_sha256")
    if raw_hash == final_hash:
        return "run_mpecss"
    bnlp_hash = (bnlp_summary or {}).get("point_sha256")
    if bnlp_hash and bnlp_hash == final_hash:
        return "external_bnlp"
    return "external_lpec_bstat"


def _certificate_rank(result: Optional[Dict[str, Any]]) -> int:
    if not result:
        return 0
    details = result.get("bstat_details") or {}
    classification = details.get("classification")
    if classification in {"B-stationary", "B-stationary (S + LICQ)", "B-stationary (final push)", "not B-stationary"}:
        return 2
    if classification in {"uncertified_favorable", "uncertified_descent_found"}:
        return 1
    return 0


def _preserve_stronger_raw_certificate(
    raw_res: Optional[Dict[str, Any]],
    final_res: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    # Do not let postprocessing erase a complete raw certificate with a weaker one.
    if not raw_res or not final_res:
        return final_res

    if _certificate_rank(raw_res) <= _certificate_rank(final_res):
        return final_res

    merged = dict(final_res)
    for key in [
        "z_final",
        "f_final",
        "objective",
        "comp_res",
        "kkt_res",
        "stationarity",
        "status",
        "sign_test_pass",
        "b_stationarity",
        "lpec_obj",
        "licq_holds",
        "bstat_details",
    ]:
        if key in raw_res:
            merged[key] = copy.deepcopy(raw_res.get(key))

    merged["preserved_raw_certificate"] = True
    merged["preserved_raw_certificate_reason"] = (
        "postprocess_point_weaker_than_complete_raw_certificate"
    )
    return merged
