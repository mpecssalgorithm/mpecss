# Copilot instructions for `mpecss`

## Build, test, and lint commands

- Install package in editable mode (used by Kaggle notebooks): `python -m pip install -e .`
- Install with test extra (pytest dependency): `python -m pip install -e ".[test]"`
- Run preflight checks before benchmark runs: `python mpecss/helpers/preflight_checks.py`
- Run a benchmark suite:  
  `python kaggle_setup/resumable_benchmark.py --dataset mpeclib --repo-dir . --path benchmarks/mpeclib/mpeclib-json --workers 2 --timeout 3600`
- Run a single benchmark problem (closest equivalent to a single test in this repo):  
  `python kaggle_setup/resumable_benchmark.py --dataset mpeclib --repo-dir . --path benchmarks/mpeclib/mpeclib-json --problem bard1 --workers 1 --timeout 3600`
- Quick smoke run with one instance:  
  `python kaggle_setup/resumable_benchmark.py --dataset mpeclib --repo-dir . --path benchmarks/mpeclib/mpeclib-json --num-problems 1 --workers 1`

Current repository state: there is no committed `test_*.py` unit-test suite and no configured lint tool (ruff/flake8/black/mypy) in this tree.

## High-level architecture

- **Primary solver API**: `mpecss.run_mpecss` (re-exported from `mpecss.phase_2.mpecss`).
- **Three-phase solve flow**:
  1. **Phase I** (`phase_1/feasibility.py`): multi-attempt feasibility search with optional deterministic multistart.
  2. **Phase II** (`phase_2/mpecss.py` + `phase_2/t_update.py` + `phase_2/sign_test.py`): homotopy loop with adaptive `t_k`, stationarity checks, and solver fallback handling.
  3. **Phase III** (`phase_3/*`): B-stationarity certification and polishing (`bnlp_polish`, `lpec_refinement_loop`, `bstat_post_check`).
- **Benchmark orchestration**:
  - `kaggle_setup/resumable_benchmark.py` is the wrapper/CLI used by notebooks and forwards to:
  - `mpecss.helpers.benchmark_utils.run_benchmark_main`, which runs each problem in an isolated subprocess, enforces per-problem timeout, records wide CSV rows, and writes audit artifacts.
  - In benchmark mode, raw `run_mpecss` output is followed by external postprocessing (`bnlp_polish -> lpec_refine -> bstat_post_check`) before final row export.
- **Problem loaders** (`mpecss/helpers/loaders/*.py`) normalize MPECLib/MacMPEC/NOSBENCH into a common problem dictionary consumed by all phases.
- **Solver infrastructure** (`helpers/solver_wrapper.py`, `helpers/solver_ipopt.py`, `helpers/solver_cache.py`) centralizes IPOPT/SQP selection, fallback chains, and cache/memory behavior.

## Key conventions specific to this codebase

- **Problem dictionary contract is cross-phase**: loaders must provide `x0_fn`, `build_casadi`, `G_fn`, `H_fn`, bounds, and metadata used later by multiplier extraction and stationarity checks.
- **Do not invent complementarity metrics ad hoc**. Use `mpecss.helpers.comp_residuals`:
  - `homotopy_comp_res` for Phase II stopping,
  - `biactive_residual` for stationarity/biactive logic,
  - `benchmark_feas_res` for benchmark reporting/postprocessing.
- **Keep multiplier-layout metadata aligned**: `build_casadi()` outputs (`n_orig_con`, `n_bounded_G`, `off_G_lb`, `off_H_lb`, `off_comp`, `_bounded_G_idx`) are consumed by `helpers.utils.extract_multipliers`; mismatches silently corrupt sign tests.
- **Nonstandard complementarity bounds are first-class**: preserve and propagate `G_is_free`, `lbG_eff`, `lbH_eff`, `ubH_finite`, `ubG_finite`, and `unsupported_model_reason` across loaders and phases.
- **Benchmark result schema is compatibility-sensitive**: `helpers/benchmark_utils.py` uses `OFFICIAL_COLUMNS` and audit JSON/CSV artifacts; when adding solver outputs, wire them into row mapping and audit summaries consistently.
- **Memory behavior is intentional**: caches are process-local and not thread-safe; benchmark workers rely on explicit cache clearing (`clear_solver_cache`, `clear_jacobian_cache`) between problems.
- **Data availability expectation**: benchmark JSON suites are expected under `benchmarks/*/*-json`; some dataset folders in this repo contain metadata/readme only, so runs often depend on external dataset population (local or Kaggle input mounts).
