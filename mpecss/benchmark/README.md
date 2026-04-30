# `mpecss/benchmark/` — Benchmark Orchestration

This subpackage provides the infrastructure for running large-scale benchmark campaigns across hundreds of MPEC problems.

## Modules

| Module | Description |
|---|---|
| `benchmark_utils.py` | `run_benchmark_main()` — CLI entry point for batch benchmark runs with parallel workers, resume support, and CSV output |
| `benchmark_results.py` | Result post-processing: diagnostic columns, raw vs. final comparison, point-quality evaluation |
| `benchmark_failure.py` | Failure-mode handling: OOM, crash, timeout, and load-failure result construction |
| `benchmark_audit.py` | `_BenchmarkAuditRecorder` — JSON audit trail for provenance tracking |

## Architecture

```
CLI (resumable_benchmark.py)
  └─→ run_benchmark_main()
       ├─→ Discover JSON problem files
       ├─→ Handle resume from previous CSV
       ├─→ Dispatch to parallel workers (multiprocessing)
       │    └─→ run_single_problem_internal()
       │         ├─→ Load problem (via loader_fn)
       │         ├─→ run_mpecss() [Phase I + II + III]
       │         ├─→ bnlp_polish() [external post-processing]
       │         ├─→ lpec_refinement_loop() [optional refinement]
       │         ├─→ bstat_post_check() [final certification]
       │         └─→ Write result row + audit JSON
       └─→ Write summary CSV
```

## Output Format

Each benchmark run produces a wide-format CSV with 200+ columns. Key column groups:

- **Problem metadata**: `problem_file`, `n_x`, `n_comp`, `family`
- **Configuration**: `cfg_t0`, `cfg_kappa`, `cfg_eps_tol`, etc.
- **Final results**: `status`, `stationarity`, `comp_res`, `b_stationarity`
- **Phase-level timing**: `time_phase_i`, `time_phase_ii`, `time_bnlp`, `time_lpec`
- **Audit trail**: `audit_schema_version`, `audit_pipeline`, `audit_final_source`
- **Raw vs. processed**: `raw_status`, `raw_comp_res` (pre-postprocessing values)

## Memory Management

The benchmark runner includes memory-pressure handling:
- Periodic logging of RSS and cache statistics
- Aggressive cleanup when memory exceeds 6 GB
- Per-problem solver cache clearing to prevent accumulation

See the parent [`mpecss/README.md`](../README.md) for full documentation.
