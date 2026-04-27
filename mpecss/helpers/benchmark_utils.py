# The "Marathon Runner": Managing large-scale tests.

import os
import gc
import time
import copy
import inspect
import queue as _queue_module
import logging
import argparse
import signal
import multiprocessing
import atexit
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Tuple

import numpy as np
import pandas as pd
import sys
import platform
import subprocess

try:
    import psutil
except ImportError:
    psutil = None


from mpecss.phase_2.mpecss import run_mpecss, DEFAULT_PARAMS
from mpecss.helpers.utils import IterationLog, export_csv

from mpecss.phase_3.bnlp_polish import bnlp_polish
from mpecss.phase_3.lpec_refine import lpec_refinement_loop
from mpecss.phase_3.bstationarity import bstat_post_check

from mpecss.helpers.benchmark_audit import (
    _BenchmarkAuditRecorder,
    _artifact_paths,
    _read_audit_artifact,
    _write_result_row_artifact,
    _read_result_row_artifact,
    _mark_audit_terminal_status,
    _json_safe,
)
from mpecss.helpers.benchmark_results import (
    _summarize_result_state,
    _build_point_diagnostic_evaluator,
    _apply_raw_summary_columns,
    _apply_point_diagnostic_columns,
    map_iteration_to_snapshot,
    _infer_final_result_source,
    _preserve_stronger_raw_certificate,
)
from mpecss.helpers.benchmark_failure import (
    _classify_problem_size,
    _build_failure_result,
)

logger = logging.getLogger("mpecss.benchmark")

MEMORY_LOG_INTERVAL = 10
MEMORY_AGGRESSIVE_CLEANUP_MB = 6000  # 6GB

_active_manager = None
_problems_since_memory_log = 0


def _cleanup_manager():
    global _active_manager
    if _active_manager is not None:
        try:
            _active_manager.shutdown()
        except Exception:
            pass
        _active_manager = None


def _sigterm_handler(signum, frame):
    _cleanup_manager()
    raise SystemExit(128 + signum)


atexit.register(_cleanup_manager)

if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, _sigterm_handler)

OFFICIAL_COLUMNS = [
    "benchmark_suite", "problem_file", "run_timestamp", "seed", "wall_timeout_cfg", "problem_name",
    "n_x", "n_comp", "n_con", "n_p", "family", "problem_size_mode",
    "cfg_t0", "cfg_kappa", "cfg_eps_tol", "cfg_delta_policy", "cfg_delta_k", "cfg_delta_factor",
    "cfg_delta0", "cfg_kappa_delta", "cfg_tau", "cfg_sta_tol", "cfg_max_outer", "cfg_max_restoration",
    "cfg_restoration_strategy", "cfg_perturb_eps", "cfg_gamma", "cfg_step_size", "cfg_smoothing",
    "cfg_adaptive_t", "cfg_steering", "cfg_stagnation_window", "cfg_adaptive_ipopt_tol",
    "cfg_feasibility_phase", "cfg_bstat_check", "cfg_lpec_refine", "cfg_fb_auto_retry",
    "cfg_solver_fallback", "cfg_skip_redundant_postsolve", "cfg_early_stag_window",
    "cfg_early_stag_threshold", "cfg_early_stag_floor", "cfg_k1_max_nlp_calls", "cfg_max_stag_recoveries",
    "status", "stationarity", "f_final", "comp_res", "kkt_res", "sign_test_pass", "b_stationarity",
    "lpec_obj", "licq_holds", "n_outer_iters", "n_restorations", "cpu_time_total",
    "fb_auto_retry_triggered",
    "phase_i_ran", "phase_i_success", "phase_i_cpu_time", "phase_i_ipopt_iter_count", "phase_i_n_attempts",
    "phase_i_initial_comp_res", "phase_i_final_comp_res", "phase_i_residual_improvement_pct",
    "phase_i_best_obj_regime", "phase_i_attempt_0_comp_res", "phase_i_attempt_1_comp_res",
    "phase_i_attempt_2_comp_res", "phase_i_n_restarts_attempted", "phase_i_n_restarts_rejected",
    "phase_i_best_restart_idx", "phase_i_multistart_improved", "phase_i_displacement_from_z0",
    "phase_i_unbounded_dims_count", "phase_i_interior_push_frac", "phase_i_feasibility_achieved",
    "phase_i_near_feasibility", "phase_i_skipped_large",
    "bootstrap_time", "bootstrap_iters", "final_t_k", "n_biactive_final", "n_sign_test_fails",
    "total_nlp_iters", "tracking_count_final", "stagnation_count_final", "last_feasible_t",
    "infeasibility_hits", "max_consecutive_fails_reached",
    "regime_superlinear_count", "regime_fast_count", "regime_slow_count", "regime_adaptive_jump_count",
    "regime_post_stagnation_count",
    "restoration_random_perturb_count", "restoration_directional_escape_count",
    "restoration_quadratic_reg_count", "restoration_qr_failed_count",
    "solver_ipopt_iters"
]

for pfx in ["iter1_", "last_iter_", "best_"]:
    OFFICIAL_COLUMNS += [
        pfx + "t_k", pfx + "delta_k", pfx + "comp_res", pfx + "kkt_res", pfx + "objective",
        pfx + "sign_test", pfx + "solver_status", pfx + "n_biactive", pfx + "nlp_iters",
        pfx + "solver_type", pfx + "warmstart", pfx + "t_update_regime", pfx + "cpu_time",
        pfx + "sta_tol", pfx + "improvement_ratio", pfx + "stagnation_count", pfx + "tracking_count",
        pfx + "is_tracking", pfx + "solver_fallback", pfx + "consec_fails", pfx + "best_comp_so_far",
        pfx + "best_iter_achieved", pfx + "ipopt_tol_used", pfx + "restoration_used",
        pfx + "restoration_trigger", pfx + "restoration_success", pfx + "biactive_indices",
        pfx + "lambda_G_min", pfx + "lambda_G_max", pfx + "lambda_H_min", pfx + "lambda_H_max"
    ]

OFFICIAL_COLUMNS += ["best_iter_number"]
OFFICIAL_COLUMNS += ["lambda_G_min_final", "lambda_G_max_final", "lambda_H_min_final", "lambda_H_max_final"]

OFFICIAL_COLUMNS += [
    "bnlp_ran", "bnlp_accepted", "bnlp_status", "bnlp_success", "bnlp_f_val", "bnlp_original_f_val",
    "bnlp_improvement", "bnlp_comp_res_polish", "bnlp_cpu_time", "bnlp_I1_size", "bnlp_I2_size",
    "bnlp_biactive_size", "bnlp_alt_partition_used", "bnlp_n_partitions_tried", "bnlp_phase_time",
    "bnlp_ultra_tight_ran", "bnlp_active_set_frac",
    "lpec_refine_ran", "lpec_refine_bstat_found", "lpec_refine_n_outer", "lpec_refine_n_inner_total",
    "lpec_refine_n_bnlps", "lpec_refine_n_lpecs", "lpec_refine_improvement", "lpec_refine_cpu_time",
    "lpec_phase_time",
    "bstat_cert_ran", "bstat_lpec_status", "bstat_classification", "bstat_lpec_obj", "bstat_n_biactive",
    "bstat_n_active_G", "bstat_n_active_H", "bstat_licq_rank", "bstat_licq_holds", "bstat_licq_details",
    "bstat_n_branches_total", "bstat_n_branches_explored", "bstat_n_feasible_branches", "bstat_timed_out",
    "bstat_elapsed_s", "bstat_used_relaxation", "bstat_trivial_no_biactive"
]

OFFICIAL_COLUMNS += ["time_phase_i", "time_bootstrap", "time_phase_ii", "time_bnlp", "time_lpec", "time_total", "error_msg"]

OFFICIAL_COLUMNS += [
    "audit_schema_version", "audit_pipeline", "audit_cpu_time_semantics",
    "audit_postprocess_applied", "audit_final_source", "audit_raw_result_available",
    "audit_effective_internal_timeout_s", "audit_effective_external_timeout_s",
    "audit_iteration_log_path", "audit_iteration_log_rows", "audit_iteration_log_empty",
    "audit_json_path", "audit_result_row_path",
    "audit_failure_last_phase", "audit_failure_elapsed_wall_s", "audit_failure_best_comp_res",
    "audit_failure_last_iter", "audit_failure_last_status",
    "raw_status", "raw_stationarity", "raw_f_final", "raw_comp_res", "raw_kkt_res",
    "raw_sign_test_pass", "raw_b_stationarity", "raw_lpec_obj", "raw_licq_holds",
    "raw_n_outer_iters", "raw_n_restorations", "raw_cpu_time_total",
    "raw_time_phase_i", "raw_time_phase_ii", "raw_time_total",
    "raw_bstat_lpec_status", "raw_bstat_classification",
    "raw_benchmark_feas_res", "raw_biactive_res", "raw_orig_constr_violation",
    "raw_var_bound_violation", "raw_comp_side_violation", "raw_overall_primal_violation",
    "raw_objective_eval", "raw_objective_abs_diff", "raw_point_sha256",
    "final_benchmark_feas_res", "final_biactive_res", "final_orig_constr_violation",
    "final_var_bound_violation", "final_comp_side_violation", "final_overall_primal_violation",
    "final_objective_eval", "final_objective_abs_diff", "final_point_sha256",
]


def _timeout_handler(signum, frame):
    # Signal handler for timeout.
    raise TimeoutError("Wall clock timeout exceeded")


def _get_memory_mb() -> float:
    # Get current process memory in MB.
    if psutil is not None:
        try:
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            pass
    return 0.0


def _check_and_cleanup_memory(problem_idx: int, force: bool = False):
    # Check memory usage and trigger cleanup if needed.
    global _problems_since_memory_log

    _problems_since_memory_log += 1
    current_mb = _get_memory_mb()

    if _problems_since_memory_log >= MEMORY_LOG_INTERVAL or force:
        _problems_since_memory_log = 0
        try:
            from mpecss.helpers.solver_cache import log_cache_stats, get_cache_stats
            stats = get_cache_stats()
            logger.info(
                f"[Problem #{problem_idx}] Memory: {current_mb:.0f}MB | "
                f"Caches: template={stats['template']['size']}, "
                f"solver={stats['solver']['size']}, "
                f"parametric={stats['parametric']['size']} | "
                f"Evictions: {stats['solver']['evictions'] + stats['parametric']['evictions']}"
            )
        except Exception:
            logger.info(f"[Problem #{problem_idx}] Memory: {current_mb:.0f}MB")

    if current_mb > MEMORY_AGGRESSIVE_CLEANUP_MB:
        logger.warning(
            f"Memory pressure detected: {current_mb:.0f}MB > {MEMORY_AGGRESSIVE_CLEANUP_MB}MB. "
            f"Triggering aggressive cleanup."
        )
        try:
            from mpecss.helpers.solver_cache import clear_solver_cache
            from mpecss.phase_3.bstationarity import clear_jacobian_cache as clear_bstat_jac
            clear_solver_cache(aggressive=True)
            clear_bstat_jac()
            gc.collect()
            gc.collect()  # Second pass for cyclic refs
            new_mb = _get_memory_mb()
            logger.info(f"Aggressive cleanup complete. Memory: {current_mb:.0f}MB → {new_mb:.0f}MB")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


def run_single_problem_internal(
    loader_fn: Callable[[str], Dict[str, Any]],
    problem_path: str,
    seed: int,
    tag: str,
    results_dir: str,
    save_logs: bool,
    dataset_tag: str,
    run_id: str,
    wall_timeout: Optional[float] = None,
    problem_idx: int = 0,
    custom_params: Optional[Dict[str, Any]] = None,
):
    # Core logic to run a single problem and return the wide data row.
    import gc

    from mpecss.helpers.solver_cache import clear_solver_cache, check_memory_pressure
    from mpecss.phase_3.bstationarity import clear_jacobian_cache as clear_bstat_jac

    clear_solver_cache(aggressive=False)
    clear_bstat_jac()
    gc.collect()

    if check_memory_pressure():
        clear_solver_cache(aggressive=True)
        gc.collect()

    problem_file = os.path.basename(problem_path)
    artifacts = _artifact_paths(results_dir, dataset_tag, tag, run_id, problem_file)
    audit = _BenchmarkAuditRecorder(results_dir, dataset_tag, tag, run_id, problem_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_wall = time.time()
    start_total = time.perf_counter()
    audit.update_progress("load_started", force=True, status="running")

    problem = None
    try:
        problem = loader_fn(problem_path)
        audit.set_problem_metadata(problem)
        audit.update_progress("load_completed", force=True, status="running")
    except Exception as e:
        logger.error(f"Failed to load {problem_path}: {e}")
        error_msg = f"Load error: {e}"
        audit.fail("load_failed", error_msg, "load_failed")
        return _build_failure_result(
            loader_fn=loader_fn,
            problem_dir=os.path.dirname(problem_path),
            problem_file=problem_file,
            dataset_tag=dataset_tag,
            status="load_failed",
            error_msg=error_msg,
            seed=seed,
            wall_timeout=wall_timeout,
            elapsed_wall_s=time.perf_counter() - start_total,
            run_started_at=start_wall,
            audit_json_path=artifacts["audit_json"],
        )

    params = {
        "seed": seed,
        "max_outer": 3000,  # 3000 outer iters; t_k floor prevents the iteration treadmill
        "adaptive_t": True,
        "solver_opts": {"max_iter": 5000, "tol": 1e-9},
        "feasibility_phase": True,
        "max_restorations": 50,
        "restoration_stag_window": 8,
        "progress_callback": audit.progress_callback,
    }
    if custom_params:
        for key, value in custom_params.items():
            if key == "progress_callback":
                continue
            if value is not None:
                params[key] = value
    if wall_timeout is not None:
        params["wall_timeout"] = wall_timeout * 0.80


    res = None
    raw_res = None
    after_bnlp_res = None
    time_phase_ii = 0.0
    time_bnlp = 0.0
    time_lpec = 0.0
    raw_total_time = 0.0
    try:
        z0 = problem["x0_fn"](seed)
        start_run_mpecss = time.perf_counter()
        audit.update_progress(
            "run_mpecss_started",
            force=True,
            status="running",
            seed=seed,
            wall_timeout_cfg=wall_timeout,
            internal_timeout_s=params.get("wall_timeout"),
        )
        res = run_mpecss(problem, z0, params)
        raw_total_time = time.perf_counter() - start_run_mpecss
        raw_res = copy.deepcopy(res)
        raw_summary = _summarize_result_state(raw_res)
        audit.attach_stage_summary("raw_run_mpecss", raw_summary, force=True)
        _phase_i_cpu = (res.get("phase_i_result") or {}).get("cpu_time", 0.0) or 0.0
        time_phase_ii = max(0.0, raw_total_time - _phase_i_cpu)
        audit.update_progress(
            "run_mpecss_completed",
            force=True,
            status=res.get("status"),
            best_comp_res=(raw_summary or {}).get("comp_res"),
            iteration=(raw_summary or {}).get("n_outer_iters"),
        )

        if res.get("status") == "unsupported_model":
            audit.attach_stage_summary("final", raw_summary, force=True)
            audit.update_progress(
                "postprocess_skipped_unsupported_model",
                force=True,
                status=res.get("status"),
            )
        else:
            eps_tol = float(params.get("eps_tol", DEFAULT_PARAMS.get("eps_tol", 1e-6)))
            time_bnlp_start = time.perf_counter()
            audit.update_progress("external_bnlp_started", force=True, status="running")
            res = bnlp_polish(res, problem, eps_tol=eps_tol)
            time_bnlp = time.perf_counter() - time_bnlp_start
            after_bnlp_res = copy.deepcopy(res)
            audit.attach_stage_summary("after_external_bnlp", _summarize_result_state(after_bnlp_res), force=True)
            audit.update_progress(
                "external_bnlp_completed",
                force=True,
                status=res.get("status"),
                bnlp_accepted=(res.get("bnlp_polish") or {}).get("accepted"),
            )

            time_lpec_start = time.perf_counter()
            audit.update_progress("external_lpec_bstat_started", force=True, status="running")
            lpec_refine_params = {
                "tol_B": max(1e-10, min(1e-8, eps_tol)),
                "tol_comp": max(1e-8, eps_tol),
                "rho_init": 0.01,
            }
            res = _invoke_lpec_refinement_loop(res, problem, params=lpec_refine_params)
            res = bstat_post_check(res, problem, eps_tol=eps_tol)
            res = _preserve_stronger_raw_certificate(raw_res, res)
            time_lpec = time.perf_counter() - time_lpec_start
            audit.attach_stage_summary("final", _summarize_result_state(res), force=True)
            audit.update_progress("external_lpec_bstat_completed", force=True, status=res.get("status"))

    except MemoryError as e:
        logger.error(f"OOM for {os.path.basename(problem_path)}: {e}")
        clear_solver_cache()
        clear_bstat_jac()
        gc.collect()
        error_msg = f"MemoryError: {e}"
        audit.fail("oom", error_msg, audit.payload.get("last_phase", "run_mpecss"))
        return _build_failure_result(
            loader_fn=loader_fn,
            problem_dir=os.path.dirname(problem_path),
            problem_file=problem_file,
            dataset_tag=dataset_tag,
            status="oom",
            error_msg=error_msg,
            seed=seed,
            wall_timeout=wall_timeout,
            elapsed_wall_s=time.perf_counter() - start_total,
            run_started_at=start_wall,
            problem_metadata=problem,
            audit_json_path=artifacts["audit_json"],
            audit_info=_read_audit_artifact(artifacts["audit_json"]),
        )
    except Exception as e:
        err_str = str(e)
        _OOM_SIGNALS = (
            "bad_alloc",
            "std::bad_alloc",
            "failed to map segment",
            "cannot allocate memory",
            "out of memory",
        )
        if any(sig in err_str.lower() for sig in _OOM_SIGNALS):
            logger.error(f"OOM (CasADi/system) for {os.path.basename(problem_path)}: {err_str[:200]}")
            clear_solver_cache()
            clear_bstat_jac()
            gc.collect()
            error_msg = f"OOM: {err_str[:300]}"
            audit.fail("oom", error_msg, audit.payload.get("last_phase", "run_mpecss"))
            return _build_failure_result(
                loader_fn=loader_fn,
                problem_dir=os.path.dirname(problem_path),
                problem_file=problem_file,
                dataset_tag=dataset_tag,
                status="oom",
                error_msg=error_msg,
                seed=seed,
                wall_timeout=wall_timeout,
                elapsed_wall_s=time.perf_counter() - start_total,
                run_started_at=start_wall,
                problem_metadata=problem,
                audit_json_path=artifacts["audit_json"],
                audit_info=_read_audit_artifact(artifacts["audit_json"]),
            )
        logger.error(f"Solver error for {os.path.basename(problem_path)}: {err_str[:300]}")
        clear_solver_cache()
        clear_bstat_jac()
        gc.collect()
        error_msg = f"Solver error: {err_str[:300]}"
        audit.fail("crashed", error_msg, audit.payload.get("last_phase", "run_mpecss"))
        return _build_failure_result(
            loader_fn=loader_fn,
            problem_dir=os.path.dirname(problem_path),
            problem_file=problem_file,
            dataset_tag=dataset_tag,
            status="crashed",
            error_msg=error_msg,
            seed=seed,
            wall_timeout=wall_timeout,
            elapsed_wall_s=time.perf_counter() - start_total,
            run_started_at=start_wall,
            problem_metadata=problem,
            audit_json_path=artifacts["audit_json"],
            audit_info=_read_audit_artifact(artifacts["audit_json"]),
        )

    total_time = time.perf_counter() - start_total

    row = {col: None for col in OFFICIAL_COLUMNS}

    row["benchmark_suite"] = dataset_tag
    row["problem_file"]    = problem_file
    row["run_timestamp"]   = timestamp
    row["seed"]            = seed
    row["wall_timeout_cfg"] = wall_timeout        # FIX #8a: was always None
    row["problem_name"]    = problem.get("name", "unknown")
    n_x = problem.get("n_x", 0)
    row["n_x"]             = n_x
    row["n_comp"]          = problem.get("n_comp", 0)
    row["n_con"]           = problem.get("n_con", 0)
    row["n_p"]             = problem.get("n_p", 0)
    row["family"]          = problem.get("family", "")
    row["problem_size_mode"] = _classify_problem_size(n_x)  # FIX #8b: was always None

    for k, v in DEFAULT_PARAMS.items():
        row[f"cfg_{k}"] = v
    for k, v in params.items():
        if k != "solver_opts":
            row[f"cfg_{k}"] = v

    row["status"]                = res.get("status")
    row["stationarity"]          = res.get("stationarity")
    row["f_final"]               = res.get("f_final")
    row["comp_res"]              = res.get("comp_res")
    row["kkt_res"]               = res.get("kkt_res")
    row["sign_test_pass"]        = res.get("sign_test_pass")
    row["b_stationarity"]        = res.get("b_stationarity")
    row["lpec_obj"]              = res.get("lpec_obj")
    row["licq_holds"]            = res.get("licq_holds")
    row["n_outer_iters"]         = res.get("n_outer_iters")
    row["n_restorations"]        = res.get("n_restorations")
    row["cpu_time_total"]        = total_time
    row["fb_auto_retry_triggered"]  = res.get("fb_auto_retry_triggered")
    row["audit_schema_version"] = 1
    row["audit_pipeline"] = "run_mpecss+external_bnlp+lpec_refine+bstat_post_check"
    row["audit_cpu_time_semantics"] = "wall_clock_perf_counter"
    row["audit_effective_internal_timeout_s"] = params.get("wall_timeout")
    row["audit_effective_external_timeout_s"] = wall_timeout
    row["audit_json_path"] = artifacts["audit_json"]
    row["audit_result_row_path"] = artifacts["result_row_json"]
    row["audit_raw_result_available"] = raw_res is not None

    p1 = res.get("phase_i_result", {})
    if p1:
        row["phase_i_ran"] = True
        for k in [
            "success", "cpu_time", "ipopt_iter_count", "n_attempts",
            "initial_comp_res", "final_comp_res", "residual_improvement_pct",
            "best_obj_regime", "attempt_0_comp_res", "attempt_1_comp_res",
            "attempt_2_comp_res", "n_restarts_attempted", "n_restarts_rejected",
            "best_restart_idx", "multistart_improved", "displacement_from_z0",
            "unbounded_dims_count", "interior_push_frac",
            "feasibility_achieved", "near_feasibility",
        ]:
            row[f"phase_i_{k}"] = p1.get(k)
        row["time_phase_i"]          = p1.get("cpu_time", 0)
        row["phase_i_skipped_large"] = (p1.get("solver_status") == "skipped_large")

    logs = res.get("logs", [])
    if logs:
        regimes = [l.t_update_regime for l in logs]
        row["regime_superlinear_count"]        = regimes.count("superlinear")
        row["regime_fast_count"]               = regimes.count("fast")
        row["regime_slow_count"]               = regimes.count("slow")
        row["regime_adaptive_jump_count"]      = regimes.count("adaptive_jump")
        row["regime_post_stagnation_count"]    = regimes.count("post_stagnation_fast")
        row["total_nlp_iters"]                 = sum(l.nlp_iter_count for l in logs)
        row["final_t_k"]                       = logs[-1].t_k
        row["n_biactive_final"]                = logs[-1].n_biactive
        row["n_sign_test_fails"]               = sum(1 for l in logs if l.sign_test == "FAIL")
        row["tracking_count_final"]            = logs[-1].tracking_count
        row["stagnation_count_final"]          = logs[-1].stagnation_count

        row.update(map_iteration_to_snapshot(logs[0],  "iter1_"))
        row.update(map_iteration_to_snapshot(logs[-1], "last_iter_"))
        best_log = min(logs, key=lambda l: l.comp_res)
        row.update(map_iteration_to_snapshot(best_log, "best_"))
        row["best_iter_number"]      = best_log.iteration

        final_comp = res.get("comp_res")
        final_obj = res.get("f_final")
        if final_comp is not None and np.isfinite(final_comp) and final_comp < best_log.comp_res:
            row["best_comp_res"] = final_comp
            row["best_objective"] = final_obj
            row["best_sign_test"] = (
                "PASS" if res.get("sign_test_pass") is True
                else "FAIL" if res.get("sign_test_pass") is False
                else None
            )
            row["best_solver_status"] = "phase_iii_final"
            row["best_iter_number"] = None

        row["lambda_G_min_final"]    = logs[-1].lambda_G_min
        row["lambda_G_max_final"]    = logs[-1].lambda_G_max
        row["lambda_H_min_final"]    = logs[-1].lambda_H_min
        row["lambda_H_max_final"]    = logs[-1].lambda_H_max
    row["audit_iteration_log_rows"] = len(logs)
    row["audit_iteration_log_empty"] = len(logs) == 0

    for k in [
        "bootstrap_time", "bootstrap_iters", "last_feasible_t",
        "infeasibility_hits", "max_consecutive_fails_reached",
        "restoration_random_perturb_count", "restoration_directional_escape_count",
        "restoration_quadratic_reg_count", "restoration_qr_failed_count",
        "solver_ipopt_iters",
    ]:
        row[k] = res.get(k)

    bnlp = res.get("bnlp_polish", {})
    if bnlp:
        row["bnlp_ran"] = True
        for k in [
            "accepted",           # → bnlp_accepted
            "status",             # → bnlp_status
            "success",            # → bnlp_success
            "f_val",              # → bnlp_f_val
            "original_f_val",     # → bnlp_original_f_val
            "improvement",        # → bnlp_improvement
            "comp_res_polish",    # → bnlp_comp_res_polish
            "cpu_time",           # → bnlp_cpu_time
            "alt_partition_used", # → bnlp_alt_partition_used
            "n_partitions_tried", # → bnlp_n_partitions_tried
            "ultra_tight_ran",    # → bnlp_ultra_tight_ran
            "active_set_frac",    # → bnlp_active_set_frac
        ]:
            row[f"bnlp_{k}"] = bnlp.get(k)
        row["bnlp_I1_size"]      = len(bnlp.get("I1", []))
        row["bnlp_I2_size"]      = len(bnlp.get("I2", []))
        row["bnlp_biactive_size"] = len(bnlp.get("I_biactive", []))
        row["time_bnlp"]         = time_bnlp
        row["bnlp_phase_time"]   = time_bnlp   # FIX #8c: was always None

    lpec = res.get("lpec_refine", {})
    if lpec:
        row["lpec_refine_ran"] = True
        for k in ["bstat_found", "n_outer", "n_inner_total", "n_bnlps", "n_lpecs", "improvement", "cpu_time"]:
            row[f"lpec_refine_{k}"] = lpec.get(k)
        row["time_lpec"]         = time_lpec
        row["lpec_phase_time"]   = time_lpec   # FIX #8d: was always None

    bstat = res.get("bstat_details", {})
    if bstat:
        row["bstat_cert_ran"] = True
        for k in [
            "lpec_status", "classification", "lpec_obj", "n_biactive",
            "n_active_G", "n_active_H", "licq_rank", "licq_holds", "licq_details",
            "n_branches_total", "n_branches_explored", "n_feasible_branches",
            "timed_out", "elapsed_s", "used_relaxation", "trivial_no_biactive",
        ]:
            row[f"bstat_{k}"] = bstat.get(k)

    row["time_phase_ii"]  = time_phase_ii
    row["time_bootstrap"] = res.get("bootstrap_time")   # FIX #8e: was always None
    row["time_total"]     = total_time

    if save_logs:
        log_path = artifacts["iteration_log"]
        export_csv(logs, log_path)
        row["audit_iteration_log_path"] = log_path
        audit.attach_artifact("phase_ii_iteration_log_csv", log_path, force=True)

    raw_summary = _summarize_result_state(raw_res)
    after_bnlp_summary = _summarize_result_state(after_bnlp_res)
    final_summary = _summarize_result_state(res)
    _apply_raw_summary_columns(row, raw_summary)
    row["raw_time_phase_ii"] = time_phase_ii
    row["raw_time_total"] = raw_total_time
    row["audit_final_source"] = _infer_final_result_source(raw_summary, after_bnlp_summary, final_summary)
    row["audit_postprocess_applied"] = row["audit_final_source"] not in (None, "run_mpecss")

    diagnostic_eval = None
    try:
        diagnostic_eval = _build_point_diagnostic_evaluator(problem)
    except Exception as exc:
        audit.update_progress("diagnostics_unavailable", force=True, status="running", reason=str(exc)[:200])

    raw_diagnostics = diagnostic_eval(raw_res.get("z_final")) if diagnostic_eval and raw_res else None
    final_diagnostics = diagnostic_eval(res.get("z_final")) if diagnostic_eval else None
    _apply_point_diagnostic_columns(row, "raw", raw_diagnostics, (raw_res or {}).get("f_final"))
    _apply_point_diagnostic_columns(row, "final", final_diagnostics, res.get("f_final"))
    audit.attach_diagnostics("raw", raw_diagnostics, force=True)
    audit.attach_diagnostics("final", final_diagnostics, force=True)
    audit.complete(res.get("status", "unknown"), final_summary)
    _write_result_row_artifact(row, artifacts["result_row_json"])

    from mpecss.helpers.solver_cache import clear_solver_cache
    from mpecss.phase_3.bstationarity import clear_jacobian_cache as clear_bstat_jac
    clear_solver_cache()
    clear_bstat_jac()
    if 'logs' in res:
        res['logs'] = None  # Free the logs list (can be large)
    gc.collect()

    return row


def run_benchmark_main(loader_fn: Callable[[str], Dict[str, Any]], dataset_tag: str, default_path: str, results_dir_override: str = None):
    # Entry point for the three main benchmark runner scripts.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True,
    )
    
    parser = argparse.ArgumentParser(description=f"Parallel {dataset_tag} Benchmark Runner")
    parser.add_argument("--tag",          type=str,   default="Official")
    parser.add_argument("--problem",      type=str,   help="Problem name or substring filter")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--workers",      type=int,   default=2,
                        help="Number of parallel workers. Each worker runs one problem at a time (default: 2). "
                             "Recommended: 2 for 7-8GB RAM systems. Use 1 if experiencing OOM errors.")
    parser.add_argument("--timeout",      type=float, default=3600.0,
                        help="Per-problem wall-clock timeout in seconds (default: 3600). "
                             "Set 0 to disable.")
    parser.add_argument("--mem-limit-gb", type=float, default=None,
                        help="Soft per-worker RAM cap in GB (Linux-based Kaggle runtimes only). "
                             "When omitted (default), every problem is free to use "
                             "as much memory as the OS will allocate; each problem "
                             "runs in its own isolated process so one OOM-killed "
                             "problem cannot affect any other worker. "
                             "Example: --mem-limit-gb 4.0")
    parser.add_argument("--save-logs",    action="store_true", help="Save detailed per-iteration CSV logs")
    parser.add_argument("--sort-by-size", action="store_true", help="Sort problems by file size (small -> large)")
    parser.add_argument("--shuffle",      action="store_true", default=True, 
                        help="Shuffle problems randomly to distribute RAM load evenly (default: True, use --no-shuffle to disable)")
    parser.add_argument("--no-shuffle",   dest="shuffle", action="store_false", 
                        help="Disable shuffling (process problems alphabetically)")
    parser.add_argument("--path",         type=str,   default=default_path,
                        help="Path to benchmark JSON directory")
    parser.add_argument("--problem-list", type=str,   default=None,
                        help="Path to a text file listing problem filenames (one per line). "
                             "Lines starting with '#' are ignored. Use this to run a subset of problems.")
    parser.add_argument("--num-problems", type=int,   default=None,
                        help="Limit to first N problems (useful for quick official test runs, e.g., --num-problems 10)")
    parser.add_argument("--resume",       type=str,   help="Path to existing CSV results to resume from")
    parser.add_argument("--retry-failed", action="store_true", help="When resuming, ignore past OOM/timeout/crash results and re-run them")
    parser.add_argument("--solver-params-json", type=str, default=None,
                        help="JSON object with solver-parameter overrides (for example: '{\"t0\": 0.1, \"adaptive_t\": false}')")
    parser.add_argument("--output-dir",   type=str,   default=None,
                        help="Directory to save results (default: ./results). "
                             "For Kaggle runs, prefer /kaggle/working/outputs so artifacts persist.")
    args = parser.parse_args()

    custom_solver_params: Dict[str, Any] = {}
    if args.solver_params_json:
        try:
            parsed_params = json.loads(args.solver_params_json)
        except Exception as exc:
            raise ValueError(f"Invalid --solver-params-json value: {exc}") from exc
        if not isinstance(parsed_params, dict):
            raise ValueError("--solver-params-json must decode to a JSON object.")
        custom_solver_params = parsed_params
    args.solver_params = custom_solver_params

    if args.timeout is not None and args.timeout <= 0:
        args.timeout = None

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    if results_dir_override:
        results_dir = os.path.abspath(results_dir_override)
    elif args.output_dir:
        results_dir = os.path.abspath(args.output_dir)
    else:
        results_dir = os.path.abspath("results")
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {results_dir}")

    if not os.path.isdir(args.path):
        logger.error(f"Benchmark path not found: {args.path}")
        return

    problem_files = [f for f in os.listdir(args.path) if f.endswith(".json")]

    if getattr(args, 'problem_list', None):
        if not os.path.isfile(args.problem_list):
            logger.error(f"Problem list file not found: {args.problem_list}")
            return
        with open(args.problem_list, 'r', encoding='utf-8') as f:
            allowed_problems = set()
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    allowed_problems.add(line)
        original_count = len(problem_files)
        problem_files = [f for f in problem_files if f in allowed_problems]
        logger.info(f"Filtered by problem list ({args.problem_list}): {len(problem_files)} of {original_count} problems selected.")

    if args.sort_by_size:
        problem_files.sort(key=lambda f: os.path.getsize(os.path.join(args.path, f)))
        logger.info("Problem execution order: Sorted by size (small -> large).")
    elif getattr(args, 'shuffle', False):
        import random
        random.seed(args.seed)
        random.shuffle(problem_files)
        logger.info(f"Problem execution order: Shuffled randomly (seed={args.seed}) to distribute RAM load.")
    else:
        problem_files.sort()
        logger.info("Problem execution order: Alphabetical.")


    all_results: List[Dict[str, Any]] = []
    if args.resume:
        if not os.path.isfile(args.resume):
            logger.error(f"Resume file not found: {args.resume}")
            return
        
        try:
            df_old = pd.read_csv(args.resume)
            if getattr(args, 'retry_failed', False):
                failed_mask = df_old['status'].isin(['oom', 'timeout', 'crashed', 'Exception', 'load_failed'])
                df_success = df_old[~failed_mask]
                all_results = df_success.to_dict('records')
                done_files = set(df_success['problem_file'].tolist())
            else:
                all_results = df_old.to_dict('records')
                done_files = set(df_old['problem_file'].tolist())
                
            count_before = len(problem_files)
            problem_files = [f for f in problem_files if f not in done_files]
            logger.info(f"Resuming from {args.resume}: skipped {count_before - len(problem_files)} already completed problems.")
        except Exception as e:
            logger.error(f"Failed to read resume file {args.resume}: {e}")
            return

    if args.problem:
        problem_files = [f for f in problem_files if args.problem in f]

    if args.num_problems is not None and args.num_problems > 0:
        original_count = len(problem_files)
        problem_files = problem_files[:args.num_problems]
        logger.info(f"Limiting to {args.num_problems} problems (reduced from {original_count})")

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_path = os.path.join(results_dir, f"{dataset_tag}_full_{args.tag}_{timestamp}.csv")
    if args.resume:
        summary_path = os.path.join(results_dir, f"{dataset_tag}_full_{args.tag}_{timestamp}_resumed.csv")

    if psutil:
        vm = psutil.virtual_memory()
        avail_gb = vm.available / 1024**3
        total_gb = vm.total / 1024**3
        cap_note = (
            f"per-worker cap: {args.mem_limit_gb:.1f} GB (Linux-based runtime only)"
            if getattr(args, "mem_limit_gb", None)
            else "no per-problem cap — each problem may use all available memory"
        )
        logger.info(
            f"System memory: {avail_gb:.1f} GB currently free / {total_gb:.1f} GB total "
            f"({cap_note}). Each problem runs in an isolated process — "
            f"one failure cannot affect other workers."
        )

    logger.info(
        f"Starting {dataset_tag} benchmark: {len(problem_files)} problem(s), "
        f"{args.workers} worker(s), timeout={args.timeout}s."
    )
    logger.info(f"Results will be written to: {summary_path}")

    env_path = _write_run_env(
        results_dir,
        timestamp,
        dataset_tag,
        args,
        summary_path=summary_path,
        problem_files=problem_files,
        benchmark_status="started",
    )

    all_results = _run_parallel_isolated(
        problem_files, loader_fn, args, results_dir, dataset_tag, summary_path, timestamp,
        custom_params=custom_solver_params,
    )

    _write_run_env(
        results_dir,
        timestamp,
        dataset_tag,
        args,
        summary_path=summary_path,
        problem_files=problem_files,
        benchmark_status="completed",
        result_count=len(all_results),
        env_path=env_path,
    )
    logger.info(f"Benchmark complete. Results: {summary_path}")


def _write_run_env(
    results_dir: str,
    timestamp: str,
    dataset_tag: str,
    args,
    summary_path: Optional[str] = None,
    problem_files: Optional[List[str]] = None,
    benchmark_status: str = "completed",
    result_count: Optional[int] = None,
    env_path: Optional[str] = None,
) -> str:
    # Write a machine-readable JSON snapshot of every setting that could affect
    env = {
        "run_timestamp":  timestamp,
        "dataset_tag":    dataset_tag,
        "cwd":            os.getcwd(),
        "benchmark_status": benchmark_status,
        "cli_args": {
            "tag":          args.tag,
            "problem":      args.problem,
            "seed":         args.seed,
            "workers":      args.workers,
            "timeout_s":    args.timeout,
            "mem_limit_gb": getattr(args, "mem_limit_gb", None),
            "path":         args.path,
            "save_logs":    args.save_logs,
            "sort_by_size": getattr(args, "sort_by_size", False),
            "shuffle":      getattr(args, "shuffle", False),
            "num_problems": getattr(args, "num_problems", None),
            "resume":       getattr(args, "resume", None),
            "retry_failed": getattr(args, "retry_failed", False),
            "solver_params_json": getattr(args, "solver_params_json", None),
            "solver_params": getattr(args, "solver_params", None),
        },
        "reproducibility": {
            "effective_external_timeout_s": args.timeout,
            "effective_internal_timeout_s": (args.timeout * 0.80) if args.timeout else None,
            "cpu_time_semantics": "wall_clock_perf_counter",
            "timing_comparable_for_literature": bool(args.workers == 1),
            "windows_mem_limit_effective": platform.system().lower() != "windows",
        },
        "problem_selection": {
            "problem_count": len(problem_files or []),
            "problem_files": problem_files or [],
        },
        "result_artifacts": {
            "summary_csv": summary_path,
            "audit_trace_dir": os.path.join(results_dir, "audit_traces"),
            "iteration_log_dir": os.path.join(results_dir, "iteration_logs"),
            "result_count": result_count,
        },
        "env_vars": {
            k: os.environ.get(k, "not set")
            for k in [
                "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            ]
        },
        "python": {
            "version":      platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable":   sys.executable,
        },
        "platform": {
            "system":   platform.system(),
            "release":  platform.release(),
            "machine":  platform.machine(),
            "node":     platform.node(),
        },
        "packages": {},
        "hardware": {},
        "module_paths": {
            "benchmark_utils": __file__,
        },
    }

    project_root_guess = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env["module_paths"]["project_root_guess"] = project_root_guess

    try:
        import mpecss
        env["module_paths"]["mpecss_init"] = mpecss.__file__
    except Exception:
        env["module_paths"]["mpecss_init"] = "unknown"

    for pkg in ["casadi", "numpy", "pandas", "scipy", "psutil", "matplotlib"]:
        try:
            import importlib.metadata
            env["packages"][pkg] = importlib.metadata.version(pkg)
        except Exception:
            env["packages"][pkg] = "unknown"

    try:
        import psutil
        vm = psutil.virtual_memory()
        env["hardware"]["ram_total_gb"]     = round(vm.total / 1024**3, 2)
        env["hardware"]["ram_available_gb"] = round(vm.available / 1024**3, 2)
        env["hardware"]["cpu_logical"]      = psutil.cpu_count(logical=True)
        env["hardware"]["cpu_physical"]     = psutil.cpu_count(logical=False)
    except Exception:
        pass

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    env["hardware"]["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    env["git_root"] = "unknown"
    env["git_commit"] = "unknown"
    env["git"] = {
        "root": "unknown",
        "commit": "unknown",
        "commit_full": "unknown",
        "branch": "unknown",
        "dirty": None,
        "dirty_excluding_results": None,
        "status_porcelain": [],
        "code_status_porcelain": [],
        "diff_sha256": None,
        "diff_shortstat": None,
    }
    for candidate in [project_root_guess, os.getcwd(), os.path.dirname(__file__)]:
        try:
            git_root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=candidate,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=git_root,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            commit_full = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=git_root,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=git_root,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            status_text = subprocess.check_output(
                ["git", "status", "--porcelain=v1", "--untracked-files=all"],
                cwd=git_root,
                stderr=subprocess.DEVNULL,
            ).decode()
            status_lines = [line.rstrip() for line in status_text.splitlines() if line.strip()]
            code_status_lines = [
                line for line in status_lines
                if not re.search(r"(^|[\\/])results([\\/]|$)", line[3:])
            ]
            diff_bytes = subprocess.check_output(
                ["git", "diff", "--binary", "HEAD"],
                cwd=git_root,
                stderr=subprocess.DEVNULL,
            )
            diff_shortstat = subprocess.check_output(
                ["git", "diff", "--shortstat", "HEAD"],
                cwd=git_root,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            env["git_root"] = git_root
            env["git_commit"] = commit
            env["git"] = {
                "root": git_root,
                "commit": commit,
                "commit_full": commit_full,
                "branch": branch,
                "dirty": bool(status_lines),
                "dirty_excluding_results": bool(code_status_lines),
                "status_porcelain": status_lines,
                "code_status_porcelain": code_status_lines,
                "diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
                "diff_shortstat": diff_shortstat,
            }
            break
        except Exception:
            continue

    env_path = env_path or os.path.join(
        results_dir, f"{dataset_tag}_run_env_{args.tag}_{timestamp}.json"
    )
    try:
        with open(env_path, "w") as f:
            json.dump(env, f, indent=2)
        logger.info(f"Run environment snapshot: {env_path}")
    except Exception as e:
        logger.warning(f"Could not write run environment snapshot: {e}")
    return env_path


def _worker_process(problem_file, loader_fn, args_path, seed, tag, results_dir,
                    save_logs, dataset_tag, run_id, timeout, mem_limit_gb, result_queue,
                    custom_params=None):
    # The "Solo Runner": Executing one specific problem.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    if mem_limit_gb and mem_limit_gb > 0:
        try:
            import resource
            limit_bytes = int(mem_limit_gb * 1024 ** 3)
            headroom = int(0.512 * 1024 ** 3)
            cap = limit_bytes + headroom
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        except (ImportError, ValueError, resource.error):
            pass
    
    res = None
    worker_start = time.time()
    audit_json_path = _artifact_paths(results_dir, dataset_tag, tag, run_id, problem_file)["audit_json"]
    try:
        res = run_single_problem_internal(
            loader_fn, os.path.join(args_path, problem_file),
            seed, tag, results_dir, save_logs, dataset_tag, run_id, timeout,
            problem_idx=0,  # Worker process runs one problem at a time
            custom_params=custom_params,
        )
    except MemoryError:
        res = _build_failure_result(
            loader_fn=loader_fn,
            problem_dir=args_path,
            problem_file=problem_file,
            dataset_tag=dataset_tag,
            status="oom",
            error_msg="MemoryError: worker exceeded memory limit",
            seed=seed,
            wall_timeout=timeout,
            run_started_at=worker_start,
            elapsed_wall_s=time.time() - worker_start,
            audit_json_path=audit_json_path,
            audit_info=_read_audit_artifact(audit_json_path),
        )
    except BaseException as e:   # includes KeyboardInterrupt, SystemExit, etc.
        res = _build_failure_result(
            loader_fn=loader_fn,
            problem_dir=args_path,
            problem_file=problem_file,
            dataset_tag=dataset_tag,
            status="crashed",
            error_msg=f"Worker error: {type(e).__name__}: {e}",
            seed=seed,
            wall_timeout=timeout,
            run_started_at=worker_start,
            elapsed_wall_s=time.time() - worker_start,
            audit_json_path=audit_json_path,
            audit_info=_read_audit_artifact(audit_json_path),
        )
    finally:
        try:
            from mpecss.helpers.solver_cache import clear_solver_cache
            from mpecss.phase_3.bstationarity import clear_jacobian_cache as clear_bstat_jac
            clear_solver_cache(aggressive=True)
            clear_bstat_jac()
            gc.collect()
            gc.collect()  # Second pass for cyclic refs
        except Exception:
            pass  # Don't let cleanup failures mask the actual error

    try:
        result_queue.put((problem_file, res))
    except Exception as qe:
        try:
            slim = {
                "problem_file": res.get("problem_file", problem_file),
                "status":       res.get("status", "crashed"),
                "error_msg":    str(res.get("error_msg", ""))[:200],
                "audit_result_row_path": res.get("audit_result_row_path"),
                "audit_json_path": res.get("audit_json_path"),
            }
            result_queue.put((problem_file, slim))
        except Exception:
            pass  # If this also fails the monitor loop will detect exit code != 0


def _run_parallel_isolated(problem_files, loader_fn, args, results_dir, dataset_tag, summary_path, run_id, custom_params=None):
    # The "Race Coordinator": Managing multiple runners at once.
    mp_context = multiprocessing.get_context('spawn')
    manager = mp_context.Manager()
    
    global _active_manager
    _active_manager = manager
    
    all_results = []
    completed = 0
    total = len(problem_files)
    benchmark_start = time.time()
    last_memory_log_time = benchmark_start  # For periodic memory logging

    remaining = list(problem_files)
    active_procs = {}  # problem_file -> (Process, start_time)
    result_queue = manager.Queue()
    
    timeout_per_problem = args.timeout if args.timeout else None

    while remaining or active_procs:
        current_time = time.time()
        if current_time - last_memory_log_time > 300:  # 5 minutes
            last_memory_log_time = current_time
            if psutil:
                vm = psutil.virtual_memory()
                avail_gb = vm.available / 1024**3
                used_pct = vm.percent
                elapsed_hrs = (current_time - benchmark_start) / 3600
                logger.info(
                    f"[Memory Check] {elapsed_hrs:.1f}h elapsed | "
                    f"Progress: {completed}/{total} ({100*completed/total:.0f}%) | "
                    f"RAM: {avail_gb:.1f}GB free ({used_pct:.0f}% used) | "
                    f"Active workers: {len(active_procs)}"
                )

        while len(active_procs) < args.workers and remaining:
            f = remaining.pop(0)
            
            p = mp_context.Process(
                target=_worker_process,
                args=(f, loader_fn, args.path, args.seed, args.tag,
                      results_dir, args.save_logs, dataset_tag, run_id, args.timeout,
                      getattr(args, "mem_limit_gb", None), result_queue, custom_params),
            )
            p.start()
            active_procs[f] = (p, time.time())

        while True:
            try:
                problem_file, res = result_queue.get(timeout=0.2)
                res = _hydrate_queue_result(problem_file, res, results_dir, dataset_tag, args.tag, run_id)
                if problem_file not in active_procs:
                    logger.warning(
                        f"Ignoring late/duplicate result for {problem_file}; "
                        "worker is no longer active"
                    )
                    continue
                dp, _ = active_procs.pop(problem_file)
                dp.join(timeout=1.0)
                completed += 1
                elapsed = time.time() - benchmark_start
                prob_time = res.get('cpu_time_total', res.get('time_total', '?'))
                if isinstance(prob_time, (int, float)):
                    prob_time = f"{prob_time:.1f}s"
                size_tag = res.get('problem_size_mode', '?')
                logger.info(
                    f"[{completed}/{total}] "
                    f"{res.get('problem_file', problem_file)} — "
                    f"{res.get('status')} | "
                    f"size={size_tag} | prob_time={prob_time} | "
                    f"elapsed={elapsed:.0f}s"
                )
                all_results.append(res)
                _save_csv(all_results, summary_path)
            except _queue_module.Empty:
                break
            except KeyboardInterrupt:
                logger.warning(
                    "\nKeyboardInterrupt received — terminating workers and "
                    "saving partial results..."
                )
                for _f, (_p, _) in list(active_procs.items()):
                    try:
                        _p.terminate()
                        _p.join(timeout=5)
                        if _p.is_alive():
                            _p.kill()
                            _p.join(timeout=2)
                    except Exception:
                        pass
                    all_results.append({
                        "problem_file": _f,
                        "status": "interrupted",
                        "error_msg": "Cancelled by KeyboardInterrupt",
                    })
                active_procs.clear()
                remaining.clear()
                _save_csv(all_results, summary_path)
                logger.info(
                    f"Partial results saved: {len(all_results)} problems → {summary_path}"
                )
                break
            except Exception as exc:
                logger.debug(f"Queue read error: {exc}")
                break

        for f in list(active_procs.keys()):
            p, start_time = active_procs[f]

            if timeout_per_problem and time.time() - start_time > timeout_per_problem:
                logger.error(f"[{completed + 1}/{total}] {f} — wall-clock deadline exceeded, terminating")
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                p.join()
                completed += 1
                audit_json_path = _artifact_paths(results_dir, dataset_tag, args.tag, run_id, f)["audit_json"]
                elapsed_wall_s = time.time() - start_time
                audit_info = _mark_audit_terminal_status(
                    audit_json_path,
                    status="timeout",
                    error_msg="Wall-clock deadline exceeded (force killed)",
                    elapsed_wall_s=elapsed_wall_s,
                ) or _read_audit_artifact(audit_json_path)
                timeout_res = _build_failure_result(
                    loader_fn=loader_fn,
                    problem_dir=args.path,
                    problem_file=f,
                    dataset_tag=dataset_tag,
                    status="timeout",
                    error_msg="Wall-clock deadline exceeded (force killed)",
                    seed=args.seed,
                    wall_timeout=args.timeout,
                    run_started_at=start_time,
                    elapsed_wall_s=elapsed_wall_s,
                    audit_json_path=audit_json_path,
                    audit_info=audit_info,
                )
                all_results.append(timeout_res)
                _save_csv(all_results, summary_path)
                del active_procs[f]
                continue

            if not p.is_alive():
                first_queue_read = True
                while True:
                    try:
                        if first_queue_read:
                            problem_file, res = result_queue.get(timeout=0.2)
                            first_queue_read = False
                        else:
                            problem_file, res = result_queue.get_nowait()
                    except _queue_module.Empty:
                        break
                    except Exception:
                        break

                    res = _hydrate_queue_result(problem_file, res, results_dir, dataset_tag, args.tag, run_id)
                    if problem_file not in active_procs:
                        logger.warning(
                            f"Ignoring late/duplicate result for {problem_file}; "
                            "worker is no longer active"
                        )
                        continue

                    dp, _ = active_procs.pop(problem_file)
                    dp.join(timeout=1.0)
                    completed += 1
                    elapsed = time.time() - benchmark_start
                    prob_time = res.get('cpu_time_total', res.get('time_total', '?'))
                    if isinstance(prob_time, (int, float)):
                        prob_time = f"{prob_time:.1f}s"
                    logger.info(
                        f"[{completed}/{total}] "
                        f"{res.get('problem_file', problem_file)} — "
                        f"{res.get('status')} | "
                        f"prob_time={prob_time} | "
                        f"elapsed={elapsed:.0f}s"
                    )
                    all_results.append(res)
                    _save_csv(all_results, summary_path)

                if f in active_procs:
                    exit_code = p.exitcode
                    completed += 1
                    if exit_code == 0:
                        logger.error(f"[{completed}/{total}] {f} — process exited cleanly but sent no result")
                        crash_status = "crashed"
                        crash_msg = "Worker exited without sending result"
                    elif exit_code in (-9, 137, 9):
                        logger.error(f"[{completed}/{total}] {f} — OOM-killed by the kernel (exit={exit_code}).")
                        crash_status = "oom"
                        crash_msg = f"OOM kill (exit {exit_code})"
                    elif exit_code in (-11, 139, 11):
                        logger.error(f"[{completed}/{total}] {f} — segmentation fault (exit={exit_code})")
                        crash_status = "crashed"
                        crash_msg = f"Segfault (exit {exit_code})"
                    else:
                        logger.error(f"[{completed}/{total}] {f} — process killed (exit={exit_code})")
                        crash_status = "crashed"
                        crash_msg = f"Process terminated with exit code {exit_code}"

                    audit_json_path = _artifact_paths(results_dir, dataset_tag, args.tag, run_id, f)["audit_json"]
                    elapsed_wall_s = time.time() - start_time
                    audit_info = _mark_audit_terminal_status(
                        audit_json_path,
                        status=crash_status,
                        error_msg=crash_msg,
                        elapsed_wall_s=elapsed_wall_s,
                    ) or _read_audit_artifact(audit_json_path)
                    crash_res = _build_failure_result(
                        loader_fn=loader_fn,
                        problem_dir=args.path,
                        problem_file=f,
                        dataset_tag=dataset_tag,
                        status=crash_status,
                        error_msg=crash_msg,
                        seed=args.seed,
                        wall_timeout=args.timeout,
                        run_started_at=start_time,
                        elapsed_wall_s=elapsed_wall_s,
                        audit_json_path=audit_json_path,
                        audit_info=audit_info,
                    )
                    all_results.append(crash_res)
                    _save_csv(all_results, summary_path)
                    del active_procs[f]
                    p.join()

    try:
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except Exception:
                break
    except Exception:
        pass  # Ignore race condition errors during cleanup

    if _active_manager is not None:
        try:
            _active_manager.shutdown()
        except Exception:
            pass
        _active_manager = None

    return all_results


def _save_csv(results: List[Dict[str, Any]], path: str) -> None:
    # Write the current results list to a CSV, keeping only OFFICIAL_COLUMNS in order.
    df   = pd.DataFrame(results)
    if "problem_file" in df.columns:
        df = df.drop_duplicates(subset=["problem_file"], keep="last")
    cols = [c for c in OFFICIAL_COLUMNS if c in df.columns]
    df[cols].to_csv(path, index=False)
