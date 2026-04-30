#!/usr/bin/env python3
"""Preflight checks for local and Kaggle benchmark runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

MIN_PYTHON = (3, 10)
REQUIRED_MODULES = ("numpy", "pandas", "scipy", "casadi", "psutil", "matplotlib")
PAPER_PYTHON = (3, 12, 12)
EXPECTED_PACKAGE_VERSIONS = {
    "casadi": "3.7.2",
    "numpy": "2.4.4",
    "pandas": "3.0.1",
    "scipy": "1.17.1",
    "psutil": "7.2.2",
    "matplotlib": "3.10.8",
}
EXPECTED_DATASETS = {
    "mpeclib": ("mpeclib", "mpeclib-json", 92),
    "macmpec": ("macmpec", "macmpec-json", 191),
    "nosbench": ("nosbench", "nosbench-json", 603),
}


def _format_version(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)


def _check_python() -> list[str]:
    current = sys.version_info[:3]
    if current < MIN_PYTHON:
        return [
            f"Python {_format_version(MIN_PYTHON)}+ is required, found {_format_version(current)}."
        ]
    return []


def _check_paper_python() -> list[str]:
    current = sys.version_info[:3]
    if current != PAPER_PYTHON:
        return [
            "Paper runs used Python "
            f"{_format_version(PAPER_PYTHON)}; current interpreter is {_format_version(current)}."
        ]
    return []


def _check_required_modules() -> list[str]:
    failures: list[str] = []
    for module_name in REQUIRED_MODULES:
        try:
            __import__(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                failures.append(f"Missing required package '{module_name}'.")
            else:
                failures.append(
                    f"Importing '{module_name}' failed because '{exc.name}' is missing."
                )
        except ImportError as exc:
            failures.append(f"Importing '{module_name}' failed: {exc}")
    return failures


def _check_pinned_versions() -> list[str]:
    warnings: list[str] = []
    for package_name, expected in EXPECTED_PACKAGE_VERSIONS.items():
        try:
            installed = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
        if installed != expected:
            warnings.append(
                f"{package_name} version differs from requirements-lock.txt: "
                f"expected {expected}, found {installed}."
            )
    return warnings


def _check_solver_plugins() -> list[str]:
    failures: list[str] = []
    try:
        import casadi as ca
    except ImportError:
        return failures

    try:
        if not ca.has_nlpsol("ipopt"):
            failures.append("CasADi IPOPT plugin is not available.")
    except Exception as exc:
        failures.append(f"Could not query CasADi IPOPT plugin: {exc}")

    try:
        if not ca.has_conic("qpoases"):
            failures.append("CasADi qpOASES conic plugin is not available.")
    except Exception as exc:
        failures.append(f"Could not query CasADi qpOASES plugin: {exc}")

    return failures


def _check_repo_layout(repo_root: Path) -> list[str]:
    failures: list[str] = []
    required_files = (
        repo_root / "pyproject.toml",
        repo_root / "kaggle_setup" / "resumable_benchmark.py",
        repo_root / "mpecss" / "benchmark" / "benchmark_utils.py",
    )
    for path in required_files:
        if not path.is_file():
            failures.append(f"Required file not found: {path}")
    return failures


def _normalize_benchmark_path(path: Path, dataset: str) -> Path | None:
    suite_dir, json_dir, _ = EXPECTED_DATASETS[dataset]
    candidates = [
        path,
        path / "benchmarks" / suite_dir / json_dir,
        path / suite_dir / json_dir,
    ]
    for candidate in candidates:
        if candidate.is_dir() and candidate.name == json_dir:
            return candidate
    return None


def _candidate_benchmark_paths(
    repo_root: Path,
    dataset: str,
    explicit_path: str | None,
) -> list[Path]:
    suite_dir, json_dir, _ = EXPECTED_DATASETS[dataset]
    candidates: list[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    env_root = os.environ.get("MPECSS_BENCHMARK_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.append(repo_root / "benchmarks" / suite_dir / json_dir)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(kaggle_input.rglob(f"benchmarks/{suite_dir}/{json_dir}"))

    normalized: list[Path] = []
    for candidate in candidates:
        resolved = _normalize_benchmark_path(candidate, dataset)
        if resolved is not None and resolved not in normalized:
            normalized.append(resolved)
    return normalized


def _check_benchmark_data(
    repo_root: Path,
    dataset: str | None,
    explicit_path: str | None,
) -> list[str]:
    failures: list[str] = []
    if dataset is None:
        return failures

    _, _, expected_count = EXPECTED_DATASETS[dataset]
    candidates = _candidate_benchmark_paths(repo_root, dataset, explicit_path)
    if not candidates:
        failures.append(
            f"Benchmark data for '{dataset}' was not found. "
            "Attach the DOI/Kaggle benchmark artifact or pass --benchmark-path."
        )
        return failures

    json_path = candidates[0]
    json_count = len(list(json_path.glob("*.json")))
    if json_count != expected_count:
        failures.append(
            f"Benchmark data for '{dataset}' has {json_count} JSON files at {json_path}; "
            f"expected {expected_count}."
        )
    return failures


def _check_output_dir(output_dir: str | None) -> list[str]:
    if not output_dir:
        return []
    path = Path(output_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return [f"Could not create output directory {path}: {exc}"]
    if not os.access(path, os.W_OK):
        return [f"No write permission for output directory {path}."]
    return []


def _check_git_state(repo_root: Path) -> list[str]:
    warnings: list[str] = []
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"Could not inspect Git state: {exc}"]

    if commit.returncode != 0:
        return ["No Git commit is available; create a release commit before DOI archiving."]

    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if dirty.returncode == 0 and dirty.stdout.strip():
        warnings.append("Git working tree is dirty; archive from a clean tagged commit.")
    return warnings


def _check_kaggle_working_dir() -> list[str]:
    kaggle_working = Path("/kaggle/working")
    if not kaggle_working.exists():
        return []
    if not os.access(kaggle_working, os.W_OK):
        return [f"No write permission for {kaggle_working}."]
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(EXPECTED_DATASETS.keys()),
        help="Benchmark suite whose data directory and expected file count must be checked.",
    )
    parser.add_argument(
        "--benchmark-path",
        help="Explicit benchmark JSON directory, benchmark root, or repository root.",
    )
    parser.add_argument("--output-dir", help="Directory where benchmark artifacts will be written.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat reproducibility warnings, including version and Git-state drift, as failures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]

    failures: list[str] = []
    warnings: list[str] = []
    failures.extend(_check_python())
    failures.extend(_check_repo_layout(repo_root))
    failures.extend(_check_required_modules())
    failures.extend(_check_solver_plugins())
    failures.extend(_check_benchmark_data(repo_root, args.dataset, args.benchmark_path))
    failures.extend(_check_output_dir(args.output_dir))
    failures.extend(_check_kaggle_working_dir())
    warnings.extend(_check_paper_python())
    warnings.extend(_check_pinned_versions())
    warnings.extend(_check_git_state(repo_root))

    if args.strict:
        failures.extend(warnings)
        warnings = []

    if failures:
        print("[preflight] FAILED")
        for failure in failures:
            print(f"[preflight] - {failure}")
        return 1

    if warnings:
        print("[preflight] OK with warnings")
        for warning in warnings:
            print(f"[preflight] - {warning}")
    else:
        print("[preflight] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
