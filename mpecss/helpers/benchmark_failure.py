# Failure result handling and problem classification for benchmark runs.

import os
from datetime import datetime
from typing import Dict, Any, Optional, Callable

from mpecss.helpers.benchmark_results import _apply_raw_summary_columns


def _classify_problem_size(n_x: int) -> str:
    # Derive problem_size_mode from the number of decision variables.
    if n_x < 50:
        return "small"
    if n_x < 500:
        return "medium"
    return "large"


def _build_failure_result(
    loader_fn: Callable[[str], Dict[str, Any]],
    problem_dir: str,
    problem_file: str,
    dataset_tag: str,
    status: str,
    error_msg: str,
    seed: Optional[int] = None,
    wall_timeout: Optional[float] = None,
    run_started_at: Optional[float] = None,
    elapsed_wall_s: Optional[float] = None,
    problem_metadata: Optional[Dict[str, Any]] = None,
    audit_json_path: Optional[str] = None,
    audit_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # Build a lightweight failure row enriched with problem metadata when possible.
    result: Dict[str, Any] = {
        "benchmark_suite": dataset_tag,
        "problem_file": problem_file,
        "problem_name": os.path.basename(problem_file).replace(".nl.json", ""),
        "seed": seed,
        "status": status,
        "error_msg": error_msg,
        "wall_timeout_cfg": wall_timeout,
        "audit_schema_version": 1,
        "audit_pipeline": "run_mpecss+external_bnlp+lpec_refine+bstat_post_check",
        "audit_cpu_time_semantics": "wall_clock_perf_counter",
        "audit_json_path": audit_json_path,
        "audit_result_row_path": None,
        "audit_failure_last_phase": (audit_info or {}).get("last_phase"),
        "audit_effective_internal_timeout_s": (wall_timeout * 0.80) if wall_timeout else None,
        "audit_effective_external_timeout_s": wall_timeout,
    }
    if run_started_at is not None:
        result["run_timestamp"] = datetime.fromtimestamp(run_started_at).strftime("%Y%m%d_%H%M%S")
    if elapsed_wall_s is not None:
        result["time_total"] = elapsed_wall_s
        result["cpu_time_total"] = elapsed_wall_s
        result["audit_failure_elapsed_wall_s"] = elapsed_wall_s

    raw_summary = ((audit_info or {}).get("stage_summaries") or {}).get("raw_run_mpecss")
    if raw_summary:
        _apply_raw_summary_columns(result, raw_summary)
        result["audit_raw_result_available"] = True
    progress = (audit_info or {}).get("progress") or {}
    result["audit_failure_best_comp_res"] = progress.get("best_comp_res")
    result["audit_failure_last_iter"] = progress.get("iteration")
    result["audit_failure_last_status"] = progress.get("solver_status") or progress.get("status")

    try:
        problem = problem_metadata or loader_fn(os.path.join(problem_dir, problem_file))
        n_x = int(problem.get("n_x", 0))
        result.update(
            {
                "problem_name": problem.get("name", result["problem_name"]),
                "n_x": n_x,
                "n_comp": problem.get("n_comp", 0),
                "n_con": problem.get("n_con", 0),
                "n_p": problem.get("n_p", 0),
                "family": problem.get("family", ""),
                "problem_size_mode": _classify_problem_size(n_x),
            }
        )
    except Exception:
        pass
    return result
