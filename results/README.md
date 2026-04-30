# `results/` — Benchmark Output Directory

This directory is the default output location for benchmark results when running MPECSS locally (i.e., outside of Kaggle). On Kaggle, outputs are written to `/kaggle/working/outputs/` instead.

---

## What Gets Written Here

When a benchmark run completes (via `mpecss/benchmark/benchmark_utils.py` or `kaggle_setup/resumable_benchmark.py`), the following files are produced:

### 1. Summary CSV (Wide Format)

```
<dataset_tag>_full_<tag>_<YYYYMMDD_HHMMSS>.csv
```

Example: `mpeclib_full_Official_20260428_143012.csv`

This is a single CSV file containing one row per benchmark problem. Each row has 200+ columns capturing:

| Column Group | Examples | Description |
|---|---|---|
| **Problem metadata** | `problem_file`, `problem_name`, `n_x`, `n_comp`, `family` | Problem dimensions and identifiers |
| **Solver configuration** | `cfg_t0`, `cfg_kappa`, `cfg_eps_tol`, `cfg_adaptive_t` | All solver parameters used for this run |
| **Final results** | `status`, `stationarity`, `f_final`, `comp_res`, `b_stationarity` | Convergence status and solution quality |
| **Phase I details** | `phase_i_ran`, `phase_i_success`, `phase_i_final_comp_res` | Feasibility phase diagnostics |
| **Phase II details** | `n_outer_iters`, `final_t_k`, `n_sign_test_fails` | Homotopy loop diagnostics |
| **Phase III details** | `bstat_classification`, `bstat_lpec_obj`, `licq_holds` | B-stationarity certification details |
| **Timing** | `time_phase_i`, `time_phase_ii`, `time_bnlp`, `time_lpec`, `time_total` | Per-phase wall-clock timing |
| **Audit trail** | `audit_schema_version`, `audit_pipeline`, `audit_final_source` | Provenance tracking columns |
| **Raw vs. final** | `raw_status`, `raw_comp_res`, `final_point_sha256` | Pre- and post-processing comparison |

### 2. Per-Problem Iteration Logs (Optional)

When `--save-logs` is enabled, each problem produces:

```
<dataset_tag>_<tag>_<run_id>_<problem_file>_iterations.csv
```

These contain one row per Phase II outer iteration with columns: `iteration`, `t_k`, `comp_res`, `kkt_res`, `sign_test`, `n_biactive`, `solver_status`, `cpu_time`, etc.

### 3. Audit JSON Artifacts (Per-Problem)

```
<dataset_tag>_<tag>_<run_id>_<problem_file>_audit.json
```

Machine-readable audit trail recording every stage transition, intermediate results, and timing for each problem.

### 4. Result Row JSON (Per-Problem)

```
<dataset_tag>_<tag>_<run_id>_<problem_file>_result.json
```

The final result row for each problem in JSON format, useful for programmatic inspection.

### 5. ZIP Archive (Kaggle Only)

When running on Kaggle with `--bundle-output`, all outputs are bundled into:

```
<dataset_tag>_full_<tag>_<YYYYMMDD_HHMMSS>_artifacts.zip
```

---

## Interpreting Results

### Status Values

| Status | Meaning |
|---|---|
| `converged` | Solver converged to a stationary point within tolerance |
| `max_iter` | Maximum Phase II iterations reached without convergence |
| `nlp_failure` | Underlying NLP solver (IPOPT/SQP) returned a non-success status |
| `timeout` | Wall-clock budget exhausted |
| `stagnation` | Adaptive jump limit reached without progress |
| `stationarity_unverifiable` | Complementarity satisfied but B-stationarity could not be certified |
| `unsupported_model` | Problem structure not supported by current implementation |
| `oom` | Out of memory |
| `crashed` | Unexpected exception during solve |
| `load_failed` | Problem JSON file could not be loaded |

### Stationarity Values

| Stationarity | Meaning |
|---|---|
| `B` | B-stationary point certified (strongest guarantee) |
| `C` | C-stationary point (weaker; sign test or LPEC detected non-B-stationarity) |
| `FAIL` | Stationarity could not be determined |

---

## Reproducing Results

To reproduce results locally (equivalent to a Kaggle run):

```bash
# Install the package
pip install -e .

# Run MPECLib benchmark (92 problems, ~2 hours)
python kaggle_setup/resumable_benchmark.py \
    --dataset mpeclib \
    --workers 4 \
    --timeout 3600 \
    --seed 42 \
    --save-logs \
    --output-dir results/
```

To resume a partial run:

```bash
python kaggle_setup/resumable_benchmark.py \
    --dataset mpeclib \
    --workers 4 \
    --timeout 3600 \
    --seed 42 \
    --resume results/mpeclib_full_Official_20260428_143012.csv
```

---

## Release Artifact Note

This directory is intentionally empty in the Git repository. Benchmark outputs are large and machine-specific, so they are generated fresh for each run. For the DOI release, archive the official Kaggle CSVs, per-problem audit JSON files, and generated figure inputs as a separate results artifact, then record the artifact DOI/version and SHA-256 checksum in `CHECKSUMS.md`.

Before archiving, verify that the official result artifact contains:

- `mpeclib_full_Official_<timestamp>.csv`
- `macmpec_full_Official_<timestamp>.csv`
- Six NOSBENCH group CSVs, or one merged NOSBENCH CSV plus the six original group CSVs
- Audit JSON and iteration logs when `--save-logs` is enabled
