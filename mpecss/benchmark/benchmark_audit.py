# Audit recording and result artifact management for benchmark runs.

import os
import json
import time
import hashlib
import re
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger("mpecss.benchmark")


def _sanitize_artifact_component(value: str) -> str:
    text = str(value)
    text = text.replace(".nl.json", "").replace(".json", "")
    text = os.path.basename(text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._") or "artifact"


def _artifact_stem(dataset_tag: str, tag: str, run_id: str, problem_file: str) -> str:
    parts = [
        _sanitize_artifact_component(dataset_tag),
        _sanitize_artifact_component(tag),
        _sanitize_artifact_component(run_id),
        _sanitize_artifact_component(problem_file),
    ]
    return "_".join(parts)


def _artifact_paths(results_dir: str, dataset_tag: str, tag: str, run_id: str, problem_file: str) -> Dict[str, str]:
    stem = _artifact_stem(dataset_tag, tag, run_id, problem_file)
    return {
        "audit_json": os.path.join(results_dir, "audit_traces", f"{stem}.json"),
        "iteration_log": os.path.join(results_dir, "iteration_logs", f"{stem}.csv"),
        "result_row_json": os.path.join(results_dir, "row_traces", f"{stem}.json"),
    }


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    return str(value)


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2)
    os.replace(tmp_path, path)


def _point_fingerprint(z: Any) -> Dict[str, Any]:
    if z is None:
        return {}
    arr = np.asarray(z, dtype=float).flatten()
    if arr.size == 0:
        return {"point_sha256": None, "point_dim": 0, "point_inf_norm": 0.0, "point_l2_norm": 0.0}
    return {
        "point_sha256": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest(),
        "point_dim": int(arr.size),
        "point_inf_norm": float(np.linalg.norm(arr, ord=np.inf)),
        "point_l2_norm": float(np.linalg.norm(arr)),
        "point_all_finite": bool(np.all(np.isfinite(arr))),
    }


class _BenchmarkAuditRecorder:
    # Write an incremental per-problem audit artifact that survives timeouts.

    def __init__(
        self,
        results_dir: str,
        dataset_tag: str,
        tag: str,
        run_id: str,
        problem_file: str,
    ) -> None:
        self.paths = _artifact_paths(results_dir, dataset_tag, tag, run_id, problem_file)
        self._start_perf = time.perf_counter()
        self._last_flush_perf = 0.0
        self._flush_interval_s = 2.0
        self.payload: Dict[str, Any] = {
            "schema_version": 1,
            "dataset_tag": dataset_tag,
            "tag": tag,
            "run_id": run_id,
            "problem_file": problem_file,
            "status": "running",
            "last_phase": "worker_started",
            "started_at": datetime.now().isoformat(),
            "last_updated_at": datetime.now().isoformat(),
            "elapsed_wall_s": 0.0,
            "artifacts": {},
            "stage_summaries": {},
            "diagnostics": {},
            "progress": {},
        }
        self._flush(force=True)

    def _flush(self, force: bool = False) -> None:
        now = time.perf_counter()
        self.payload["elapsed_wall_s"] = now - self._start_perf
        self.payload["last_updated_at"] = datetime.now().isoformat()
        if not force and (now - self._last_flush_perf) < self._flush_interval_s:
            return
        _atomic_write_json(self.paths["audit_json"], self.payload)
        self._last_flush_perf = now

    def attach_artifact(self, key: str, path: str, force: bool = True) -> None:
        self.payload.setdefault("artifacts", {})[key] = path
        self._flush(force=force)

    def set_problem_metadata(self, problem: Dict[str, Any]) -> None:
        self.payload["problem_name"] = problem.get("name")
        self.payload["problem_metadata"] = {
            "family": problem.get("family"),
            "n_x": problem.get("n_x"),
            "n_comp": problem.get("n_comp"),
            "n_con": problem.get("n_con"),
            "n_p": problem.get("n_p"),
            "source_path": problem.get("_source_path"),
        }
        self._flush(force=True)

    def update_progress(self, phase: str, force: bool = False, status: Optional[str] = None, **fields: Any) -> None:
        self.payload["last_phase"] = phase
        if status is not None:
            self.payload["status"] = status
        self.payload.setdefault("progress", {}).update(_json_safe(fields))
        self._flush(force=force)

    def progress_callback(self, stage: str, force: bool = False, **fields: Any) -> None:
        self.update_progress(stage, force=force, **fields)

    def attach_stage_summary(self, name: str, summary: Optional[Dict[str, Any]], force: bool = True) -> None:
        if summary is not None:
            self.payload.setdefault("stage_summaries", {})[name] = _json_safe(summary)
            if summary.get("comp_res") is not None:
                self.payload.setdefault("progress", {})["best_comp_res"] = summary.get("comp_res")
        self._flush(force=force)

    def attach_diagnostics(self, name: str, diagnostics: Optional[Dict[str, Any]], force: bool = True) -> None:
        if diagnostics is not None:
            self.payload.setdefault("diagnostics", {})[name] = _json_safe(diagnostics)
        self._flush(force=force)

    def fail(self, status: str, error_msg: str, phase: str) -> None:
        self.payload["status"] = status
        self.payload["error_msg"] = error_msg
        self.payload["last_phase"] = phase
        self._flush(force=True)

    def complete(self, status: str, final_summary: Optional[Dict[str, Any]]) -> None:
        self.payload["status"] = status
        self.payload["completed_at"] = datetime.now().isoformat()
        if final_summary is not None:
            self.payload.setdefault("stage_summaries", {})["final"] = _json_safe(final_summary)
        self._flush(force=True)


def _read_audit_artifact(audit_json_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not audit_json_path or not os.path.isfile(audit_json_path):
        return None
    try:
        with open(audit_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_result_row_artifact(row: Dict[str, Any], row_json_path: Optional[str]) -> None:
    if not row_json_path:
        return
    _atomic_write_json(row_json_path, row)


def _read_result_row_artifact(row_json_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not row_json_path or not os.path.isfile(row_json_path):
        return None
    try:
        with open(row_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _mark_audit_terminal_status(
    audit_json_path: Optional[str],
    status: str,
    error_msg: Optional[str] = None,
    elapsed_wall_s: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    if not audit_json_path:
        return None
    payload = _read_audit_artifact(audit_json_path) or {"schema_version": 1}
    now = datetime.now().isoformat()
    payload["status"] = status
    payload["completed_at"] = now
    payload["last_updated_at"] = now
    if error_msg:
        payload["error_msg"] = error_msg
    if elapsed_wall_s is not None:
        payload["elapsed_wall_s"] = elapsed_wall_s
    _atomic_write_json(audit_json_path, payload)
    return payload
