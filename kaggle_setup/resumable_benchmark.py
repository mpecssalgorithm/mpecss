#!/usr/bin/env python3
"""
Kaggle-friendly resumable benchmark wrapper.

Thin wrapper over mpecss.benchmark.benchmark_utils.run_benchmark_main.
Adds Kaggle-specific features:
  --resume-latest   Find the most recent CSV under --resume-search-dir or /kaggle/input
  --resume-csv      Resume from an explicit CSV path (for example from /kaggle/input)
  --resume-search-dir  Search this directory for the latest resume CSV
  --summary-only    Print a progress summary without running any problems
  --dataset         Choose which benchmark suite to run
  --repo-dir        Path to the Org-MPECSS checkout (for Kaggle path setup)
  --skip-preflight  Skip the preflight checks

Usage from a Kaggle notebook:
    !python kaggle_setup/resumable_benchmark.py --dataset mpeclib --repo-dir /kaggle/working/mpecss-kaggle --workers 4
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("kaggle_resumable")


def _find_latest_csv(results_dir: str, dataset_tag: str) -> str | None:
    """Find the most recent CSV for a given dataset tag in the results directory."""
    pattern = os.path.join(results_dir, f"{dataset_tag}_full_*.csv")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None
    return candidates[-1]


def _find_latest_csv_recursive(search_root: str, dataset_tag: str) -> str | None:
    """Find the most recent CSV for a dataset tag anywhere under search_root."""
    pattern = os.path.join(search_root, "**", f"{dataset_tag}_full_*.csv")
    candidates = [
        path
        for path in glob.glob(pattern, recursive=True)
        if os.path.isfile(path)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (os.path.getmtime(path), path))


def _print_summary(csv_path: str, dataset_tag: str) -> None:
    """Print a quick progress summary from an existing CSV."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        total = len(df)
        status_counts = df["status"].value_counts()

        print("=" * 70)
        print(f"  MPECSS Benchmark Summary — {dataset_tag}")
        print(f"  CSV: {csv_path}")
        print("=" * 70)
        print(f"  Total problems: {total}")
        print()
        for status, count in status_counts.items():
            pct = 100 * count / total
            print(f"    {status:30s}  {count:4d}  ({pct:5.1f}%)")
        print()

        # B-stationarity summary
        if "b_stationarity" in df.columns:
            bstat = df["b_stationarity"].value_counts()
            print("  B-stationarity:")
            for val, count in bstat.items():
                print(f"    {str(val):30s}  {count:4d}")
            print()

        # Timing
        if "time_total" in df.columns:
            t = df["time_total"].dropna()
            if len(t) > 0:
                print(f"  Timing (time_total):")
                print(f"    mean:   {t.mean():.1f}s")
                print(f"    median: {t.median():.1f}s")
                print(f"    max:    {t.max():.1f}s")
                print(f"    sum:    {t.sum():.0f}s  ({t.sum()/3600:.1f}h)")
            print()

        print("=" * 70)

    except Exception as e:
        print(f"[summary] Error reading {csv_path}: {e}", file=sys.stderr)


def _has_output_artifacts(results_dir: str) -> bool:
    """Return True when the results directory contains at least one file."""
    root = Path(results_dir)
    return root.exists() and any(path.is_file() for path in root.rglob("*"))


def _sanitize_name_component(value: str) -> str:
    """Sanitize text for filesystem-friendly archive names."""
    text = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(value))
    text = text.strip("_")
    return text or "run"


def _build_bundle_base_name(
    results_dir: str,
    dataset_tag: str,
    tag: str,
    resume_csv: str | None,
    retry_failed: bool,
) -> str:
    """Build archive basename from run context."""
    latest_csv = _find_latest_csv(results_dir, dataset_tag)
    if latest_csv:
        # Keep bundle name aligned with the produced benchmark CSV.
        archive_stem = f"{Path(latest_csv).stem}_artifacts"
    else:
        mode = "retry" if retry_failed else ("resumed" if resume_csv else "fresh")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_stem = f"{dataset_tag}_full_{_sanitize_name_component(tag)}_{ts}_{mode}_artifacts"

    return str(Path(results_dir).parent / archive_stem)


def _bundle_results(
    results_dir: str,
    dataset_tag: str,
    tag: str,
    resume_csv: str | None,
    retry_failed: bool,
) -> str:
    """Create a context-aware zip archive next to the results directory."""
    archive_base = _build_bundle_base_name(
        results_dir=results_dir,
        dataset_tag=dataset_tag,
        tag=tag,
        resume_csv=resume_csv,
        retry_failed=retry_failed,
    )
    archive_path = shutil.make_archive(archive_base, "zip", root_dir=results_dir)
    print(f"[resumable] Bundled results to: {archive_path}")
    return archive_path


def _count_json_files(directory: str) -> int:
    """Count top-level JSON files in a benchmark directory."""
    if not os.path.isdir(directory):
        return 0
    return sum(
        1
        for name in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, name)) and name.endswith(".json")
    )


def _resolve_json_subdir(directory: str, dataset_tag: str) -> str | None:
    """Resolve a likely *-json subdirectory when the provided path is a parent folder."""
    preferred = os.path.join(directory, f"{dataset_tag}-json")
    if _count_json_files(preferred) > 0:
        return preferred

    candidates = []
    for name in sorted(os.listdir(directory)):
        child = os.path.join(directory, name)
        if os.path.isdir(child) and name.endswith("-json") and _count_json_files(child) > 0:
            candidates.append(child)

    if len(candidates) == 1:
        return candidates[0]
    return None


def _normalize_benchmark_json_path(path: str, dataset_tag: str) -> str:
    """Normalize benchmark path to a directory that actually contains problem JSON files."""
    if _count_json_files(path) > 0:
        return path

    resolved = _resolve_json_subdir(path, dataset_tag)
    if resolved:
        logger.warning(
            "No JSON files found directly in %s; using %s instead.",
            path,
            resolved,
        )
        return resolved
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kaggle-friendly resumable benchmark wrapper for MPECSS."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mpeclib", "macmpec", "nosbench"],
        help="Which benchmark suite to run.",
    )
    parser.add_argument(
        "--repo-dir",
        type=str,
        default=None,
        help="Path to the Org-MPECSS checkout. Added to sys.path.",
    )
    parser.add_argument("--tag", type=str, default="Official")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--problem", type=str, default=None)
    parser.add_argument("--num-problems", type=int, default=None)
    parser.add_argument("--num-problem", dest="num_problems", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--mem-limit-gb", type=float, default=None)
    parser.add_argument("--save-logs", action="store_true")
    parser.add_argument("--sort-by-size", action="store_true")
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument(
        "--solver-params-json",
        type=str,
        default=None,
        help="JSON object with solver-parameter overrides passed through to the benchmark runner.",
    )
    parser.add_argument("--path", type=str, default=None,
                        help="Override the benchmark JSON directory path.")
    parser.add_argument("--problem-list", type=str, default=None,
                        help="Path to a text file listing problem filenames (one per line). "
                             "Use this to run a subset of problems (e.g., for splitting large benchmarks).")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--resume-latest", action="store_true",
                        help="Automatically find and resume from the latest CSV under --resume-search-dir or /kaggle/input.")
    parser.add_argument("--resume-csv", type=str, default=None,
                        help="Explicit CSV path to resume from (supports /kaggle/input/... paths).")
    parser.add_argument("--resume-search-dir", type=str, default=None,
                        help="Directory to recursively search for latest CSV when --resume-latest is set.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="When resuming, re-run OOM/timeout/crash problems.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print a progress summary and exit (no solving).")
    parser.add_argument("--bundle-output", action="store_true",
                        help="Create a zip archive of the output directory after the run.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results. Defaults to /kaggle/working/outputs when available.")

    args = parser.parse_args()

    # ── Setup paths ────────────────────────────────────────────────────────
    repo_dir = args.repo_dir
    if repo_dir is None:
        # Infer from this script's location: kaggle_setup/ is inside the repo
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    # Ensure the repo is on sys.path so imports work
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    # Change to repo root so relative paths in the runners work
    os.chdir(repo_dir)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Dataset configuration ─────────────────────────────────────────────
    DATASET_CONFIG = {
        "mpeclib": {
            "loader_module": "mpecss.helpers.loaders.mpeclib_loader",
            "loader_fn_name": "load_mpeclib",
            "default_path": os.path.join(repo_dir, "benchmarks", "mpeclib", "mpeclib-json"),
        },
        "macmpec": {
            "loader_module": "mpecss.helpers.loaders.macmpec_loader",
            "loader_fn_name": "load_macmpec",
            "default_path": os.path.join(repo_dir, "benchmarks", "macmpec", "macmpec-json"),
        },
        "nosbench": {
            "loader_module": "mpecss.helpers.loaders.nosbench_loader",
            "loader_fn_name": "load_nosbench",
            "default_path": os.path.join(repo_dir, "benchmarks", "nosbench", "nosbench-json"),
        },
    }

    config = DATASET_CONFIG[args.dataset]
    if args.output_dir:
        results_dir = args.output_dir
    elif os.path.isdir("/kaggle/working"):
        results_dir = "/kaggle/working/outputs"
    else:
        results_dir = os.path.join(repo_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    print(f"[resumable] Results will be saved to: {results_dir}")

    # ── Summary-only mode ─────────────────────────────────────────────────
    if args.summary_only:
        csv_path = _find_latest_csv(results_dir, args.dataset)
        if csv_path:
            _print_summary(csv_path, args.dataset)
            return 0
        else:
            print(f"[summary] No CSV found for dataset '{args.dataset}' in {results_dir}")
            return 1

    # ── Preflight ─────────────────────────────────────────────────────────
    if not args.skip_preflight:
        import subprocess

        preflight_script = os.path.join(repo_dir, "mpecss", "helpers", "preflight_checks.py")
        legacy_preflight_script = os.path.join(repo_dir, "scripts", "preflight_checks.py")
        if os.path.isfile(preflight_script):
            rc = subprocess.run([sys.executable, preflight_script]).returncode
        elif os.path.isfile(legacy_preflight_script):
            rc = subprocess.run([sys.executable, legacy_preflight_script]).returncode
        else:
            logger.warning(
                "Preflight script not found at '%s' or legacy path '%s'",
                preflight_script,
                legacy_preflight_script,
            )
            rc = 0

        if rc != 0:
            print("[resumable] Preflight checks failed. Use --skip-preflight to override.")
            return rc

    # ── Import the loader and benchmark runner ────────────────────────────
    try:
        import importlib
        loader_mod = importlib.import_module(config["loader_module"])
        loader_fn = getattr(loader_mod, config["loader_fn_name"])
        from mpecss.benchmark.benchmark_utils import run_benchmark_main
    except ImportError as e:
        print(f"[ERROR] Could not import mpecss: {e}")
        print("Make sure the cloned repository is on sys.path and the notebook install cell completed.")
        return 1

    # ── Build sys.argv for run_benchmark_main ─────────────────────────────
    # run_benchmark_main uses argparse internally, so we inject CLI args
    # via sys.argv before calling it.
    benchmark_path = args.path or config["default_path"]

    # Smart path resolution: if the provided path doesn't exist, try Kaggle dataset mounts.
    if benchmark_path and not os.path.isdir(benchmark_path):
        found_path = None

        if os.path.isdir("/kaggle/input"):
            # Extract the benchmark-relative part (e.g., "nosbench/nosbench-json")
            parts = benchmark_path.split("/benchmarks/", 1)
            if len(parts) == 2:
                bench_relative = parts[1]  # e.g., "nosbench/nosbench-json"

                # On Kaggle: search for the actual path (handles datasets/username/... structure)
                import subprocess
                result = subprocess.run(
                    ["find", "/kaggle/input", "-type", "d", "-path", f"*/benchmarks/{bench_relative}"],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0 and result.stdout.strip():
                    found_path = result.stdout.strip().split('\n')[0]
                    logger.info(f"Found Kaggle benchmark path: {found_path}")

        if found_path:
            benchmark_path = found_path
        else:
            logger.warning(f"Benchmark path not found: {benchmark_path}")
            logger.warning("Tried searching Kaggle dataset mounts under /kaggle/input.")

    if benchmark_path and os.path.isdir(benchmark_path):
        benchmark_path = _normalize_benchmark_json_path(benchmark_path, args.dataset)

    if not benchmark_path or not os.path.isdir(benchmark_path):
        logger.error(f"Benchmark path not found: {benchmark_path}")
        logger.error("Pass a valid JSON directory with --path or use the notebook defaults.")
        return 2

    json_count = _count_json_files(benchmark_path)
    if json_count == 0:
        logger.error(f"No benchmark JSON files found in: {benchmark_path}")
        logger.error(
            "For NosBench, use the directory ending in '/benchmarks/nosbench/nosbench-json'."
        )
        return 2
    logger.info(f"Benchmark path ready: {benchmark_path} ({json_count} JSON files)")

    injected_args = [
        "resumable_benchmark",  # argv[0]
        "--tag", args.tag,
        "--workers", str(args.workers),
        "--timeout", str(args.timeout),
        "--seed", str(args.seed),
        "--path", benchmark_path,
    ]

    if args.problem:
        injected_args.extend(["--problem", args.problem])
    if args.num_problems is not None:
        injected_args.extend(["--num-problems", str(args.num_problems)])
    if args.mem_limit_gb is not None:
        injected_args.extend(["--mem-limit-gb", str(args.mem_limit_gb)])
    if args.save_logs:
        injected_args.append("--save-logs")
    if args.sort_by_size:
        injected_args.append("--sort-by-size")
    if args.shuffle:
        injected_args.append("--shuffle")
    else:
        injected_args.append("--no-shuffle")
    if args.solver_params_json:
        injected_args.extend(["--solver-params-json", args.solver_params_json])
    if args.retry_failed:
        injected_args.append("--retry-failed")
    if args.problem_list:
        problem_list_path = args.problem_list
        # Smart path resolution for problem list
        if not os.path.isfile(problem_list_path):
            # Try extracting relative path from various Kaggle working directory patterns
            for pattern in [
                "/kaggle/working/mpecss-kaggle/",
                "/kaggle/working/MPECSSCODEPAPER/",
                "/kaggle/working/",
            ]:
                if pattern in problem_list_path:
                    parts = problem_list_path.split(pattern, 1)
                    if len(parts) == 2:
                        local_path = os.path.join(repo_dir, parts[1])
                        if os.path.isfile(local_path):
                            logger.info(f"Using local problem-list path: {local_path}")
                            problem_list_path = local_path
                            break
        injected_args.extend(["--problem-list", problem_list_path])

    # Pass output directory to ensure results go to persistent location
    injected_args.extend(["--output-dir", results_dir])

    # Handle resume source selection
    resume_csv = None
    if args.resume_csv:
        if not os.path.isfile(args.resume_csv):
            logger.error(f"Resume CSV not found: {args.resume_csv}")
            return 2
        resume_csv = args.resume_csv
    elif args.resume_latest:
        search_roots = []
        if args.resume_search_dir:
            search_roots.append(args.resume_search_dir)
        # Search Kaggle input datasets so old CSV can come from uploaded outputs.
        if os.path.isdir("/kaggle/input"):
            search_roots.append("/kaggle/input")

        seen = set()
        for root in search_roots:
            root_abs = os.path.abspath(root)
            if root_abs in seen or not os.path.isdir(root_abs):
                continue
            seen.add(root_abs)
            latest = _find_latest_csv_recursive(root_abs, args.dataset)
            if latest:
                logger.info(f"Resolved resume CSV from {root_abs}: {latest}")
                resume_csv = latest
                break

        if not search_roots:
            logger.info(
                "No resume search source available. Use --resume-csv or --resume-search-dir."
            )
        if not resume_csv:
            logger.info(f"No previous CSV found for '{args.dataset}'. Starting fresh.")

    if resume_csv:
        logger.info(f"Resuming from: {resume_csv}")
        injected_args.extend(["--resume", resume_csv])

    # Replace sys.argv so run_benchmark_main's argparse sees our args
    original_argv = sys.argv
    sys.argv = injected_args
    try:
        run_benchmark_main(
            loader_fn=loader_fn,
            dataset_tag=args.dataset,
            default_path=benchmark_path,
        )
    finally:
        sys.argv = original_argv

    if args.bundle_output and not args.summary_only:
        if _has_output_artifacts(results_dir):
            _bundle_results(
                results_dir=results_dir,
                dataset_tag=args.dataset,
                tag=args.tag,
                resume_csv=resume_csv,
                retry_failed=args.retry_failed,
            )
        else:
            logger.warning(f"No output artifacts found in {results_dir}; skipping archive.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
