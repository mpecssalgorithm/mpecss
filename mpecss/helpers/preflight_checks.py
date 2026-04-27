#!/usr/bin/env python3
# Preflight checks for Kaggle benchmark notebooks.

from __future__ import annotations

import os
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
REQUIRED_MODULES = ("numpy", "pandas", "scipy", "casadi", "psutil")


def _format_version(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)


def _check_python() -> list[str]:
    current = sys.version_info[:3]
    if current < MIN_PYTHON:
        return [
            f"Python {_format_version(MIN_PYTHON)}+ is required, found {_format_version(current)}."
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


def _check_kaggle_working_dir() -> list[str]:
    kaggle_working = Path("/kaggle/working")
    if not kaggle_working.exists():
        return []
    if not os.access(kaggle_working, os.W_OK):
        return [f"No write permission for {kaggle_working}."]
    return []


def main() -> int:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]

    failures: list[str] = []
    failures.extend(_check_python())
    failures.extend(_check_repo_layout(repo_root))
    failures.extend(_check_required_modules())
    failures.extend(_check_kaggle_working_dir())

    if failures:
        print("[preflight] FAILED")
        for failure in failures:
            print(f"[preflight] - {failure}")
        return 1

    print("[preflight] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
