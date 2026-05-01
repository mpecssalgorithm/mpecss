# MPECSS: Scholtes Regularization with Adaptive Paths for Targeting B-stationary Points in MPECs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1224064767.svg)](https://doi.org/10.5281/zenodo.19949992)

---

## What is MPECSS?

MPECSS is a method for **Mathematical Programs with Equilibrium Constraints (MPECs)**, also known as **Mathematical Programs with Complementarity Constraints (MPCCs)**. These optimization problems arise in bilevel programming, game theory, nonsmooth optimal control, and engineering design. They are challenging because complementarity constraints violate standard constraint qualifications, making classical NLP approaches unreliable.

MPECSS implements a **Scholtes regularization** with an **adaptive path-following strategy** that targets **B-stationary** (Bouligand-stationary) points — the strongest first-order optimality condition available for MPECs.

### Key Features

- **Three-phase algorithm**: Feasibility search → Homotopy solver → B-stationarity certification
- **Adaptive regularization**: The regularization parameter `t_k` is updated using multiple regimes (superlinear, fast, slow, adaptive jump) based on convergence behavior
- **B-stationarity certification**: Uses LPEC (Linear Program with Equilibrium Constraints) enumeration and LICQ shortcuts to certify whether the solution is B-stationary
- **Tested on 886 problems**: MPECLib (92), MacMPEC (191), NOSBENCH (603)

---

## Quick Start

### 0. 60-Second Smoke Test

```bash
git clone https://github.com/mpecssalgorithm/mpecss.git
cd mpecss
pip install -e .
python kaggle_setup/resumable_benchmark.py \
    --dataset mpeclib \
    --problem bard1 \
    --workers 1 \
    --timeout 120 \
    --seed 42 \
    --tag Smoke \
    --save-logs \
    --output-dir results/smoke
```

Expected result: inside the `results/smoke/` directory, one CSV named `mpeclib_full_Smoke_<timestamp>.csv` containing `bard1.nl.json`, `status=converged`, `stationarity=B`, and `b_stationarity=True`.

### 1. Prerequisites

- **Python 3.10 or later** (required — the codebase uses `match`, `|` union types, and other 3.10+ features)
- **pip** (Python package manager)
- **Git** (to clone the repository)

> **Note**: No GPU is required. The solver is CPU-only. A machine with at least 8 GB RAM is recommended for benchmark runs. Tested on Linux (Kaggle), macOS, and Windows.

### 2. Clone and Install

```bash
# Clone the repository
git clone https://github.com/mpecssalgorithm/mpecss.git
cd mpecss

# Install in editable mode (recommended for development/review)
pip install -e .

# Or install with test dependencies
pip install -e ".[test]"
```

For paper reproduction, use a clean virtual environment with the pinned release dependencies instead of the loose development requirements:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows (PowerShell): .\.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
pip install -e .
```

This installs the `mpecss` package and its runtime dependencies:

| Package | Version | Purpose |
|---|---|---|
| `casadi` | ≥ 3.6.3 | Automatic differentiation and IPOPT NLP solver interface |
| `numpy` | ≥ 1.24 | Numerical array operations |
| `pandas` | ≥ 2.0 | CSV I/O for benchmark results |
| `scipy` | ≥ 1.11 | Sparse matrix operations (Jacobian evaluation in Phase III) |
| `psutil` | ≥ 5.9 | System memory monitoring during benchmark runs |
| `matplotlib` | ≥ 3.8 | Paper figure generation |

The exact package pins used for the release reproducibility environment are listed in [`requirements-lock.txt`](requirements-lock.txt).

### 3. Verify Installation

```bash
# Run preflight checks
python mpecss/helpers/preflight_checks.py

# Run unit tests
pytest tests/ -v
```

Expected output from preflight: `[preflight] OK`. If you are not using the paper environment or a clean release commit, preflight may report `[preflight] OK with warnings`; use `--strict` to make those warnings fail.

### 4. Solve a Single Problem

> **Prerequisite**: Before running this example, ensure the benchmark JSON data is present at `benchmarks/mpeclib/mpeclib-json/`. The MPECLib JSON files are included in the Git repository. For MacMPEC and NOSBENCH, download from the sources listed in [`benchmarks/README.md`](benchmarks/README.md) or use the official [Kaggle dataset](https://www.kaggle.com/datasets/mpecssalgorithm/mpecss-benchmarks).

```python
from mpecss import run_mpecss
from mpecss.helpers.loaders import load_mpeclib

# Load a problem from the MPECLib benchmark suite
problem = load_mpeclib("benchmarks/mpeclib/mpeclib-json/bard1.nl.json")

# Generate an initial point (seed controls random restarts in Phase I)
z0 = problem["x0_fn"](seed=42)

# Solve with default parameters
result = run_mpecss(problem, z0, params={"seed": 42})

# Inspect the result
print(f"Status:        {result['status']}")          # e.g., "converged"
print(f"Stationarity:  {result['stationarity']}")     # "B", "C", or "FAIL"
print(f"Comp. residual:{result['comp_res']:.3e}")     # e.g., 1.23e-09
print(f"Objective:     {result['f_final']:.6f}")      # Optimal objective value
print(f"B-stationary:  {result['b_stationarity']}")   # True/False/None
print(f"LICQ holds:    {result['licq_holds']}")       # True/False/None
print(f"CPU time:      {result['cpu_time']:.2f}s")
```

---

## Official Benchmark Workflow (Recommended for Reproducibility)

Official benchmark results are produced on the [Kaggle](https://www.kaggle.com/) platform to ensure reproducibility on standardized hardware. The complete step-by-step guide is in [`kaggle_setup/README.md`](kaggle_setup/README.md).

### What is Kaggle?

[Kaggle](https://www.kaggle.com/) is a free cloud platform that provides Jupyter notebook environments with standardized hardware (CPU, RAM, storage). We use Kaggle because:
- **Reproducibility**: Every reviewer gets the same hardware and software environment
- **Free**: No cost to create an account and run notebooks
- **No local setup**: No need to install Python, CasADi, or any dependencies locally
- **Persistent outputs**: Results are saved as downloadable artifacts

### Quick Summary of the Kaggle Workflow

1. **Create a free Kaggle account** at https://www.kaggle.com/
2. **Create a new notebook** (Click "Create" → "New Notebook")
3. **Attach the benchmark dataset**: In the right sidebar, click "Input" → "+ Add Input" → search for `mpecss-benchmarks` ([direct link](https://www.kaggle.com/datasets/mpecssalgorithm/mpecss-benchmarks)) or visit the [benchmarks folder](benchmarks/) for details.
4. **Import a notebook**: Click "File" → "Import Notebook" → select the `.ipynb` file from `kaggle_setup/`
5. **Enable Internet access**: Click "Settings" → turn Internet ON (needed for `pip install`)
6. **Run**: Click "Save Version" → name it (e.g., `MPECLib_Benchmark_Run`) → select "Save & Run All (commit)" → click "Save"
7. **Download results**: After completion, go to the "Output" tab and download the generated ZIP archive

Required benchmark data:

| Suite | Expected files | Required path |
|---|---:|---|
| MPECLib | 92 JSON files | `benchmarks/mpeclib/mpeclib-json/` |
| MacMPEC | 191 JSON files | `benchmarks/macmpec/macmpec-json/` |
| NOSBENCH | 603 JSON files | `benchmarks/nosbench/nosbench-json/` |


Full paper reproduction should be run through the Kaggle notebooks with the benchmark execution commit SHA pinned: `6117ae6aa2e118936ca2ada4c44b175d091ce8ad`. Expected output CSV names are `mpeclib_full_Official_<timestamp>.csv`, `macmpec_full_Official_<timestamp>.csv`, and one CSV for each NOSBENCH group. To produce the complete NOSBENCH table, merge the six group CSVs:

```python
import pandas as pd
import glob

nosbench_csvs = sorted(glob.glob("results/nosbench_group*_full_Official_*.csv"))
nosbench_merged = pd.concat([pd.read_csv(f) for f in nosbench_csvs], ignore_index=True)
nosbench_merged.to_csv("results/nosbench_full_Official_merged.csv", index=False)
```

### Available Notebooks

For the full notebook catalog (main benchmarks, ablation, seed-robustness, and parameter-sensitivity), see [`kaggle_setup/README.md`](kaggle_setup/README.md).

#### Main Benchmark Notebooks

| Benchmark Suite | Problems | Notebook | Expected Runtime |
|---|---|---|---|
| MPECLib | 92 | `MPECSS_Kaggle_MPECLib.ipynb` | ~2–4 hours |
| MacMPEC | 191 | `MPECSS_Kaggle_MacMPEC.ipynb` | ~4–8 hours |
| NOSBENCH Group 1 | 101 | `MPECSS_Kaggle_NosBench_Group1.ipynb` | ~4–8 hours |
| NOSBENCH Group 2 | 101 | `MPECSS_Kaggle_NosBench_Group2.ipynb` | ~4–8 hours |
| NOSBENCH Group 3 | 101 | `MPECSS_Kaggle_NosBench_Group3.ipynb` | ~4–8 hours |
| NOSBENCH Group 4 | 100 | `MPECSS_Kaggle_NosBench_Group4.ipynb` | ~4–8 hours |
| NOSBENCH Group 5 | 100 | `MPECSS_Kaggle_NosBench_Group5.ipynb` | ~4–8 hours |
| NOSBENCH Group 6 | 100 | `MPECSS_Kaggle_NosBench_Group6.ipynb` | ~4–8 hours |

#### Public Executed Kaggle Runs

| Run | Public Kaggle notebook |
|---|---|
| MPECLib official | [mpeclib-official-run](https://www.kaggle.com/code/mralexsantora/mpeclib-official-run) |
| MacMPEC official | [macmpec-official-run](https://www.kaggle.com/code/mralexsantora/macmpec-official-run) |
| NOSBENCH Group 1 official | [nosbench1-official-run](https://www.kaggle.com/code/mralexsantora/nosbench1-official-run) |
| NOSBENCH Group 2 official | [nosbench2-official-run](https://www.kaggle.com/code/mralexsantora/nosbench2-official-run) |
| NOSBENCH Group 3 official | [nosbench3-official-run](https://www.kaggle.com/code/mralexsantora/nosbench3-official-run) |
| NOSBENCH Group 4 official | [nosbench4-official-run](https://www.kaggle.com/code/mrsaurabhtanwar/nosbench4-official-run) |
| NOSBENCH Group 5 official | [nosbench5-official-run](https://www.kaggle.com/code/mrsaurabhtanwar/nosbench5-official-run) |
| NOSBENCH Group 6 official | [nosbench6-official-run](https://www.kaggle.com/code/mrsaurabhtanwar/nosbench6-official-run) |
| MacMPEC fixed Phase II ablation | [macmpec-ab-fixt](https://www.kaggle.com/code/mrsaurabhtanwar/macmpec-ab-fixt) |
| MacMPEC no Phase I ablation | [macmpec-ab-nph1](https://www.kaggle.com/code/mrsaurabhtanwar/macmpec-ab-nph1) |
| MacMPEC `kappa=0.3` | [macmpec-kappa-0-3](https://www.kaggle.com/code/mrgauravtanwar/macmpec-kappa-0-3) |
| MacMPEC `kappa=0.8` | [macmpec-kappa-0-8](https://www.kaggle.com/code/mrgauravtanwar/macmpec-kappa-0-8) |
| MacMPEC `t0=0.1` | [macmpec-t0-1](https://www.kaggle.com/code/mrgauravtanwar/macmpec-t0-1) |
| MacMPEC `t0=10` | [macmpec-t10](https://www.kaggle.com/code/mrgauravtanwar/macmpec-t10) |
| MacMPEC seed 11 | [macmpec-seed11](https://www.kaggle.com/code/mrgauravtanwar/macmpec-seed11) |
| MacMPEC seed 123 | [macmpec-seed123](https://www.kaggle.com/code/mradarshkumar/macmpec-seed123) |

> **Why is NOSBENCH split into 6 groups?** Kaggle enforces a 12-hour maximum runtime per notebook session. The 603 NOSBENCH problems cannot all be solved within that window, so they are split into 6 groups (101/101/101/100/100/100). After collecting all 6 CSV outputs, merge them for the complete NOSBENCH results table.

The main benchmark notebooks use a 3600-second per-problem timeout. MacMPEC ablation, seed-robustness, and parameter-sensitivity notebooks use 1800 seconds per problem; the paper reports those study settings separately from the main benchmark totals.

### Re-running Timed-Out or Crashed Problems

For problems that timed out or crashed in the initial run, you can re-run them with extended time budgets:

1. Upload the CSV output from the previous run as a Kaggle dataset input
2. In the notebook benchmark command, uncomment the cell `run_benchmark(resume=True, retry_failed=True)` and set the timeout to 7200 at `Configuration` cell.

This skips already-solved problems and re-attempts only the failed ones with a 2-hour per-problem budget.

Tip (Recommendation): On kaggle free tier, At the same time 5 notebooks can be run, so you can run multiple benchmark notebooks parallel.

---

## Running Benchmarks Locally (Alternative to Kaggle)

While the Kaggle workflow is recommended for reproducibility, you can also run benchmarks locally:

```bash
# Install the package
pip install -e .

# Run the full MPECLib suite (92 problems)
python kaggle_setup/resumable_benchmark.py \
    --dataset mpeclib \
    --workers 4 \
    --timeout 3600 \
    --seed 42 \
    --tag Official \
    --save-logs \
    --output-dir results/

# Run a single problem (useful for debugging)
python kaggle_setup/resumable_benchmark.py \
    --dataset mpeclib \
    --problem bard1 \
    --workers 1 \
    --timeout 3600 \
    --seed 42 \
    --save-logs

# Resume from a previous run
python kaggle_setup/resumable_benchmark.py \
    --dataset mpeclib \
    --workers 4 \
    --timeout 3600 \
    --seed 42 \
    --resume results/mpeclib_full_Official_*.csv

# Run with custom solver parameters
python kaggle_setup/resumable_benchmark.py \
    --dataset macmpec \
    --workers 4 \
    --timeout 3600 \
    --seed 42 \
    --solver-params-json '{"t0": 0.1, "adaptive_t": false}'
```

### Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | (required) | `mpeclib`, `macmpec`, or `nosbench` |
| `--workers` | 4 | Number of parallel workers (each runs one problem at a time) |
| `--timeout` | 3600 | Per-problem wall-clock timeout in seconds |
| `--seed` | 42 | Random seed for Phase I restarts |
| `--tag` | `Official` | Tag string included in output filenames |
| `--save-logs` | off | Save per-iteration CSV logs for each problem |
| `--output-dir` | `results/` | Directory for output files |
| `--problem` | (all) | Filter to problems containing this substring |
| `--num-problems` | (all) | Limit to first N problems |
| `--resume` | (none) | Path to existing CSV to resume from |
| `--retry-failed` | off | When resuming, re-run OOM/timeout/crash problems |
| `--shuffle` | on | Randomise problem order (distributes memory load) |
| `--sort-by-size` | off | Process smaller problems first |
| `--solver-params-json` | (none) | JSON object with parameter overrides |
| `--problem-list` | (none) | Text file listing specific problem filenames to run |
| `--skip-preflight` | off | Skip pre-run environment checks |

---

## Understanding the Solver Output

### Result Status

| Status | What It Means |
|---|---|
| **B-stationary** | Strongest certificate: Phase III verified B-stationarity (either via LICQ+S-stationarity equivalence, or via LPEC enumeration finding no descent direction). |
| **C-stationary** | Weaker stationarity: the complementarity tolerance was met but the point is C-stationary (not B-stationary). Still a valid stationary point. |
| **Stationarity unverifiable** | Complementarity residual is small, but the Phase III checks could not conclusively certify the stationarity class. |
| **Timeout** | The run exceeded the configured per-problem time budget (default: 3600s). |
| **Complementarity infeasibility** | MPECSS did not find a point satisfying complementarity to tolerance. This does **not** prove infeasibility of the original problem. |
| **NLP Solver Failure** | The underlying IPOPT/SQP solver returned a non-success status (e.g., restoration failure, numerical issues). |

### Output CSV Columns (Key Columns)

The output CSV contains 200+ columns. The most important ones:

| Column | Description |
|---|---|
| `status` | Final solver status (see table above) |
| `stationarity` | `B`, `C`, or `FAIL` |
| `f_final` | Objective function value at the final point |
| `comp_res` | Complementarity residual: max_i min(G_i, H_i) |
| `b_stationarity` | `True` if B-stationarity was certified, `False` if not, `None` if not checked |
| `licq_holds` | Whether MPEC-LICQ holds at the solution |
| `time_total` | Total wall-clock time in seconds |
| `n_outer_iters` | Number of Phase II outer iterations |
| `bstat_classification` | Detailed classification: `B-stationary (S + LICQ)`, `B-stationary (LPEC)`, `C-stationary`, etc. |

---

## Project Structure

```
.
├── README.md                   # This file
├── LICENSE                     # Apache 2.0 License
├── pyproject.toml              # Project metadata, dependencies, build configuration
├── requirements.txt            # Runtime dependency mirror (for pip install -r)
│
├── mpecss/                     # Core solver package (see mpecss/README.md)
│   ├── __init__.py             # Package entry point; re-exports run_mpecss()
│   ├── contracts.py            # Typed contracts (SolverStatus, ProblemSpec, SolveResult)
│   ├── phase_1/                # Phase I: Feasibility and starting point search
│   ├── phase_2/                # Phase II: Homotopy solver loop (Scholtes regularization)
│   ├── phase_3/                # Phase III: BNLP polishing and B-stationarity certification
│   ├── benchmark/              # Benchmark run orchestration and audit trail
│   └── helpers/                # Loaders, solver wrappers, monitoring, and utilities
│       ├── loaders/            # MPECLib, MacMPEC, and NOSBENCH problem loaders
│       └── solver/             # IPOPT/SQP wrappers, caching, and solver metrics
│
├── benchmarks/                 # Benchmark datasets and conversion scripts (see benchmarks/README.md)
│   ├── mpeclib/                # MPECLib: 92 GAMS-derived MPEC problems
│   ├── macmpec/                # MacMPEC: 191 AMPL-derived MPEC problems
│   └── nosbench/               # NOSBENCH: 603 nonsmooth optimal control MPECs
│
├── kaggle_setup/               # Kaggle notebooks and benchmark runner (see kaggle_setup/README.md)
│   ├── MPECSS_Kaggle_*.ipynb   # 16 Jupyter notebooks for all benchmark configurations
│   ├── resumable_benchmark.py  # CLI wrapper with resume and Kaggle-specific path handling
│   ├── study_runner.py         # Helper for ablation/sensitivity study notebooks
│   └── nosbench_splits/        # Problem lists for NOSBENCH group splitting
│
├── results/                    # Benchmark output directory (see results/README.md)
├── tests/                      # Unit test suite (see tests/README.md)
│
├── mpecss_paper/               # Manuscript LaTeX source (excluded from public repo)
├── research_papers/            # Reference literature PDFs (excluded from public repo)
└── sample_papers/              # Sample journal papers (excluded from public repo)
```

Each subdirectory contains its own `README.md` with detailed documentation. Click through to any of them for more information.

---

## The Three-Phase Algorithm (Overview)

### Phase I: Feasibility Search

Starting from an initial point `z0`, Phase I solves a minimum-complementarity NLP to find a point that approximately satisfies all constraints including complementarity. It uses multi-start with random restarts to escape poor local minima.

### Phase II: Homotopy (Scholtes Regularization)

The core solver loop. At each iteration:
1. Solve a smoothed NLP where `G(x) · H(x) ≤ t_k` replaces the hard complementarity `G(x) · H(x) = 0`
2. Evaluate the sign test: do all complementarity multipliers satisfy the S-stationarity sign conditions?
3. Update `t_k` using adaptive rules: reduce it when making progress, increase it when stuck
4. Track the best iterate (lowest complementarity residual)
5. Exit when `comp_res ≤ eps_tol` and sign test passes, or when iteration/time budgets are exhausted

### Phase III: Polishing & B-Stationarity Certification

1. **BNLP Polish**: Fix the active set and solve a "branch NLP" to clean up the solution
2. **LICQ Check**: Verify whether the constraint Jacobian has full rank at the solution. If LICQ holds and the sign test passed, B-stationarity follows by equivalence (S ⟺ B under LICQ)
3. **LPEC Enumeration**: If LICQ fails, enumerate all possible biactive partitions and solve LPECs to verify no descent direction exists

---

## Note on AI Assistance

The authors used AI-assisted code generation while developing this repository. The generated material has been reviewed and edited by the authors, who remain responsible for the implementation and results. Users are encouraged to review the code, open issues, or contribute improvements on [GitHub](https://github.com/mpecssalgorithm/mpecss).

---

## Troubleshooting

| Symptom | Check |
|---|---|
| `preflight` reports missing benchmark data | Attach the DOI/Kaggle benchmark artifact, or pass `--path`/`--benchmark-path` to the JSON directory. |
| `preflight` reports a Python or package-version warning | Recreate a clean virtual environment, install from `requirements-lock.txt`, and run `python mpecss/helpers/preflight_checks.py --strict` before archiving. |
| CasADi reports that IPOPT is unavailable | Reinstall from the pinned lock file and verify `python -c "import casadi as ca; print(ca.has_nlpsol('ipopt'))"`. |
| SQP/qpOASES checks fail | Verify `python -c "import casadi as ca; print(ca.has_conic('qpoases'))"` in the active environment. |
| Kaggle cannot clone or install the package | Turn notebook Internet access ON and make sure the `REPO_COMMIT` variable in the notebook's first code cell is set to `6117ae6aa2e118936ca2ada4c44b175d091ce8ad`. The clone cell should use a full clone, not `git clone --depth 1`. |
| Output ZIP is missing | Re-run with `--bundle-output` or download the CSV/log/audit files from `/kaggle/working/outputs/`. |

---

## Contact

- **GitHub Issues**: [mpecssalgorithm/mpecss/issues](https://github.com/mpecssalgorithm/mpecss/issues)
- **Email**: `mpecssalgorithm@gmail.com`

---

[License: Apache 2.0](LICENSE)
