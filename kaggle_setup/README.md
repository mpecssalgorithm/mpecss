# MPECSS Kaggle Benchmark Setup

This folder contains everything needed to run the MPECSS benchmarks on the [Kaggle](https://www.kaggle.com/) cloud platform. Kaggle provides free Jupyter notebook environments with standardized hardware, making it the recommended platform for reproducible benchmark runs.

---

## Why Kaggle?

- **Free**: No cost to create an account and run notebooks
- **Standardized hardware**: Every user gets the same CPU, RAM, and storage — results are comparable across runs
- **No local setup**: Python, pip, and all system libraries are pre-installed
- **Persistent outputs**: Results are saved as downloadable artifacts that persist indefinitely
- **12-hour runtime limit**: Sufficient for all benchmark suites (NOSBENCH is split into 6 groups to stay within this limit)

---

## Prerequisites

You need:
1. A **free Kaggle account** — sign up at https://www.kaggle.com/
2. A **web browser** — no software installation required
3. The **benchmark dataset** — either from Kaggle or from this GitHub repository (see Step 3 below)

---

## Step-by-Step Guide

### Step 1: Open Kaggle

Go to https://www.kaggle.com and sign in (or create a free account).

### Step 2: Create a New Notebook

1. Click **"Create"** → **"New Notebook"** in the top navigation bar
2. Or go directly to: https://www.kaggle.com/notebooks

You will see a blank Jupyter notebook in your browser.

### Step 3: Add the Benchmark Dataset

The benchmark problems (JSON files) need to be attached as a "dataset input" so the notebook can access them.

1. In the **right sidebar**, click **"Input"**
2. Click **"+ Add Input"**
3. Search for `mpecss-benchmarks` and read the [dataset instructions](https://www.kaggle.com/datasets/mpecssalgorithm/mpecss-benchmarks) download and upload or visit on [GitHub](https://github.com/mpecssalgorithm/mpecss/tree/main/benchmarks) and upload the `benchmarks/` folder as a new Kaggle dataset.

The dataset must contain the following folder structure:

```
benchmarks/
├── mpeclib/
│   └── mpeclib-json/         ← 92 .nl.json files
├── macmpec/
│   └── macmpec-json/         ← 191 .nl.json files
└── nosbench/
    └── nosbench-json/        ← 603 .json files
```

> **How does the notebook find the data?** Kaggle mounts datasets under `/kaggle/input/<dataset-name>/`. The notebooks auto-detect the benchmark JSON path by searching under `/kaggle/input/*/benchmarks/...`. The dataset name can be anything — path resolution is automatic.

### Step 4: Import a Notebook

1. Click **"File"** → **"Import Notebook"** in the notebook toolbar
2. Navigate to the `kaggle_setup/` folder in this repository
3. Select the `.ipynb` file for the benchmark you want to run (see table below)

### Step 5: Enable Internet Access

The notebook needs internet to install the `mpecss` package via pip:

1. Click **"Settings"** (gear icon in the right sidebar)
2. Toggle **"Internet"** to **ON**

> **Why is internet needed?** The notebook clones the MPECSS repository from GitHub at the pinned `REPO_COMMIT` value and installs it with `pip install -e .` This requires internet access. After installation, the benchmark itself runs entirely offline.

### Step 6: Run the Notebook

1. Click the **"Save Version"** button in the top-right corner
2. Enter a version name (e.g., `MPECLib_Benchmark_Run`)
3. Set the version type to **"Save & Run All (commit)"**
4. Click **"Save"**

This triggers a full execution of all notebook cells in sequence. The notebook will:
1. Clone the MPECSS repository from GitHub
2. Install the package with `pip install -e .`
3. Run preflight checks (Python version, dependencies, folder structure)
4. Execute the benchmark using the `resumable_benchmark.py` wrapper
5. Save results to `/kaggle/working/outputs/`
6. Create a ZIP archive of all output artifacts

> **How long does it take?** MPECLib: ~2–4 hours. MacMPEC: ~4–8 hours. Each NOSBENCH group: ~4–8 hours. You can close your browser — the notebook continues running on Kaggle's servers.

### Step 7: Download Results

After the notebook finishes:

1. Go to the **"Output"** tab (at the top of the notebook page)
2. Download the generated archive (e.g., `mpeclib_full_Official_20260428_143012_artifacts.zip`)
3. You can also browse individual files under `/kaggle/working/outputs/`

The ZIP archive contains:
- **Summary CSV**: One row per problem with 200+ columns (status, stationarity, timing, diagnostics)
- **Per-problem iteration logs** (if `--save-logs` was enabled)
- **Audit JSON artifacts**: Full provenance trail for each problem
- **Version note JSON**: Metadata about the run configuration

Tip (Recommendation): On kaggle free tier, At the same time 5 notebooks can be run, so you can run multiple benchmark notebooks parallel.

---

## Available Notebooks

### Main Benchmark Notebooks

| Benchmark Suite | Problems | Notebook | Timeout | Seed |
|---|---|---|---|---|
| MPECLib | 92 | `MPECSS_Kaggle_MPECLib.ipynb` | 3600s | 42 |
| MacMPEC | 191 | `MPECSS_Kaggle_MacMPEC.ipynb` | 3600s | 42 |
| NOSBENCH Group 1 | 101 | `MPECSS_Kaggle_NosBench_Group1.ipynb` | 3600s | 42 |
| NOSBENCH Group 2 | 101 | `MPECSS_Kaggle_NosBench_Group2.ipynb` | 3600s | 42 |
| NOSBENCH Group 3 | 101 | `MPECSS_Kaggle_NosBench_Group3.ipynb` | 3600s | 42 |
| NOSBENCH Group 4 | 100 | `MPECSS_Kaggle_NosBench_Group4.ipynb` | 3600s | 42 |
| NOSBENCH Group 5 | 100 | `MPECSS_Kaggle_NosBench_Group5.ipynb` | 3600s | 42 |
| NOSBENCH Group 6 | 100 | `MPECSS_Kaggle_NosBench_Group6.ipynb` | 3600s | 42 |

### Public Executed Kaggle Runs

All public runs below pin the benchmark execution commit `6117ae6aa2e118936ca2ada4c44b175d091ce8ad`.

| Run | Local notebook | Public Kaggle notebook |
|---|---|---|
| MPECLib official | `MPECSS_Kaggle_MPECLib.ipynb` | [mpeclib-official-run](https://www.kaggle.com/code/mralexsantora/mpeclib-official-run) |
| MacMPEC official | `MPECSS_Kaggle_MacMPEC.ipynb` | [macmpec-official-run](https://www.kaggle.com/code/mralexsantora/macmpec-official-run) |
| NOSBENCH Group 1 official | `MPECSS_Kaggle_NosBench_Group1.ipynb` | [nosbench1-official-run](https://www.kaggle.com/code/mralexsantora/nosbench1-official-run) |
| NOSBENCH Group 2 official | `MPECSS_Kaggle_NosBench_Group2.ipynb` | [nosbench2-official-run](https://www.kaggle.com/code/mralexsantora/nosbench2-official-run) |
| NOSBENCH Group 3 official | `MPECSS_Kaggle_NosBench_Group3.ipynb` | [nosbench3-official-run](https://www.kaggle.com/code/mralexsantora/nosbench3-official-run) |
| NOSBENCH Group 4 official | `MPECSS_Kaggle_NosBench_Group4.ipynb` | [nosbench4-official-run](https://www.kaggle.com/code/mrsaurabhtanwar/nosbench4-official-run) |
| NOSBENCH Group 5 official | `MPECSS_Kaggle_NosBench_Group5.ipynb` | [nosbench5-official-run](https://www.kaggle.com/code/mrsaurabhtanwar/nosbench5-official-run) |
| NOSBENCH Group 6 official | `MPECSS_Kaggle_NosBench_Group6.ipynb` | [nosbench6-official-run](https://www.kaggle.com/code/mrsaurabhtanwar/nosbench6-official-run) |
| MacMPEC fixed Phase II ablation | `MPECSS_Kaggle_MacMPEC_Ablation_FixedPhaseII.ipynb` | [macmpec-ab-fixt](https://www.kaggle.com/code/mrsaurabhtanwar/macmpec-ab-fixt) |
| MacMPEC no Phase I ablation | `MPECSS_Kaggle_MacMPEC_Ablation_NoPhaseI.ipynb` | [macmpec-ab-nph1](https://www.kaggle.com/code/mrsaurabhtanwar/macmpec-ab-nph1) |
| MacMPEC `kappa=0.3` | `MPECSS_Kaggle_MacMPEC_ParamSensitivity_kappa_0p3.ipynb` | [macmpec-kappa-0-3](https://www.kaggle.com/code/mrgauravtanwar/macmpec-kappa-0-3) |
| MacMPEC `kappa=0.8` | `MPECSS_Kaggle_MacMPEC_ParamSensitivity_kappa_0p8.ipynb` | [macmpec-kappa-0-8](https://www.kaggle.com/code/mrgauravtanwar/macmpec-kappa-0-8) |
| MacMPEC `t0=0.1` | `MPECSS_Kaggle_MacMPEC_ParamSensitivity_t0_0p1.ipynb` | [macmpec-t0-1](https://www.kaggle.com/code/mrgauravtanwar/macmpec-t0-1) |
| MacMPEC `t0=10` | `MPECSS_Kaggle_MacMPEC_ParamSensitivity_t0_10.ipynb` | [macmpec-t10](https://www.kaggle.com/code/mrgauravtanwar/macmpec-t10) |
| MacMPEC seed 11 | `MPECSS_Kaggle_MacMPEC_SeedRobustness_Seed11.ipynb` | [macmpec-seed11](https://www.kaggle.com/code/mrgauravtanwar/macmpec-seed11) |
| MacMPEC seed 123 | `MPECSS_Kaggle_MacMPEC_SeedRobustness_Seed123.ipynb` | [macmpec-seed123](https://www.kaggle.com/code/mradarshkumar/macmpec-seed123) |

### MacMPEC Ablation Study Notebooks

These run the full MacMPEC suite with a specific algorithmic component disabled:

| Study | Notebook | What Is Changed |
|---|---|---|
| No Phase I | `MPECSS_Kaggle_MacMPEC_Ablation_NoPhaseI.ipynb` | Phase I feasibility search is disabled; solver starts from `x0` directly |
| Fixed Phase II | `MPECSS_Kaggle_MacMPEC_Ablation_FixedPhaseII.ipynb` | Adaptive `t_k` update is disabled; uses fixed `t_{k+1} = κ · t_k` |


### MacMPEC Seed Robustness Notebooks

Test sensitivity to the random seed used in Phase I restarts:

| Seed | Notebook |
|---|---|
| 11 | `MPECSS_Kaggle_MacMPEC_SeedRobustness_Seed11.ipynb` |
| 42 | `MPECSS_Kaggle_MacMPEC.ipynb` (default) |
| 123 | `MPECSS_Kaggle_MacMPEC_SeedRobustness_Seed123.ipynb` |

### MacMPEC Parameter Sensitivity Notebooks

Test sensitivity to key solver parameters:

| Parameter | Value | Notebook |
|---|---|---|
| `t0` (initial regularization parameter) | 0.1 | `MPECSS_Kaggle_MacMPEC_ParamSensitivity_t0_0p1.ipynb` |
| `t0` | 1.0 | `MPECSS_Kaggle_MacMPEC.ipynb` (default) |
| `t0` | 10.0 | `MPECSS_Kaggle_MacMPEC_ParamSensitivity_t0_10.ipynb` |
| `κ` (contraction factor for t-update) | 0.3 | `MPECSS_Kaggle_MacMPEC_ParamSensitivity_kappa_0p3.ipynb` |
| `κ` | 0.5 | `MPECSS_Kaggle_MacMPEC.ipynb` (default) |
| `κ` | 0.8 | `MPECSS_Kaggle_MacMPEC_ParamSensitivity_kappa_0p8.ipynb` |

> **Note on timeouts**: All ablation, seed-robustness, and parameter-sensitivity notebooks use a **1800-second** per-problem timeout. The main benchmark notebooks (MPECLib, MacMPEC, NOSBENCH) use **3600 seconds** per problem. The paper reports study results separately from the main benchmark totals.

---

## Re-running Timed-Out or Crashed Problems

Some problems may time out or crash on the first run due to Kaggle's resource limits. To re-run only those problems with extended budgets:

1. **Upload the previous output CSV** as a new Kaggle dataset (or attach it as input to the notebook)
2. **Modify the benchmark call**: Uncomment the `run_benchmark(resume=True, retry_failed=True)` cell and comment out the original `run_benchmark()` call in the notebook. This enables resume and retry logic. Then, in the **Configuration** cell, set the timeout to `7200` seconds (2 hours per problem).

3. **Run the notebook** — it will skip already-solved problems and only re-attempt the failed ones

---

## What Happens Inside Each Notebook

Each notebook executes the following steps (you can inspect the cells):

1. **Clone the repository**:
   ```python
   !git clone https://github.com/mpecssalgorithm/mpecss.git /kaggle/working/mpecss-kaggle
   !git -C /kaggle/working/mpecss-kaggle checkout 6117ae6aa2e118936ca2ada4c44b175d091ce8ad
   ```

   > **Important**: The notebooks intentionally use a full clone followed by checkout of the benchmark execution commit. Avoid `git clone --depth 1` here because a shallow clone may not contain this older commit after newer commits are pushed.

2. **Install the package**:
   ```python
   !pip install -e /kaggle/working/mpecss-kaggle
   ```

3. **Run preflight checks**:
   ```python
   !python /kaggle/working/mpecss-kaggle/mpecss/helpers/preflight_checks.py
   ```

4. **Execute the benchmark** via `resumable_benchmark.py`:
   ```python
   !python /kaggle/working/mpecss-kaggle/kaggle_setup/resumable_benchmark.py \
       --dataset mpeclib \
       --repo-dir /kaggle/working/mpecss-kaggle \
       --workers 4 \
       --timeout 3600 \
       --seed 42 \
       --tag Official \
       --save-logs \
       --bundle-output
   ```

5. **Display summary**:
   ```python
   !python /kaggle/working/mpecss-kaggle/kaggle_setup/resumable_benchmark.py \
       --dataset mpeclib \
       --summary-only
   ```

---

## Supporting Scripts

| File | Description |
|---|---|
| `resumable_benchmark.py` | Main CLI wrapper. Handles Kaggle-specific path resolution, resume logic, result bundling, and delegates to `mpecss.benchmark.benchmark_utils.run_benchmark_main()`. |
| `study_runner.py` | Helper for ablation and sensitivity notebooks. Builds and executes the CLI command for each study configuration. |
| `nosbench_splits/` | Contains text files listing problem filenames for each NOSBENCH group (used by `--problem-list` to split the 603 problems across 6 notebooks). |

---

## Folder Structure

```
kaggle_setup/
├── README.md                                           # This file
│
├── MPECSS_Kaggle_MPECLib.ipynb                        # MPECLib benchmark (92 problems)
├── MPECSS_Kaggle_MacMPEC.ipynb                        # MacMPEC benchmark (191 problems)
│
├── MPECSS_Kaggle_MacMPEC_Ablation_NoPhaseI.ipynb      # Ablation: no Phase I
├── MPECSS_Kaggle_MacMPEC_Ablation_FixedPhaseII.ipynb  # Ablation: fixed t-update
│
├── MPECSS_Kaggle_MacMPEC_SeedRobustness_Seed11.ipynb  # Seed robustness: seed=11
├── MPECSS_Kaggle_MacMPEC_SeedRobustness_Seed123.ipynb # Seed robustness: seed=123
│
├── MPECSS_Kaggle_MacMPEC_ParamSensitivity_t0_0p1.ipynb    # Sensitivity: t0=0.1
├── MPECSS_Kaggle_MacMPEC_ParamSensitivity_t0_10.ipynb     # Sensitivity: t0=10.0
├── MPECSS_Kaggle_MacMPEC_ParamSensitivity_kappa_0p3.ipynb # Sensitivity: κ=0.3
├── MPECSS_Kaggle_MacMPEC_ParamSensitivity_kappa_0p8.ipynb # Sensitivity: κ=0.8
│
├── MPECSS_Kaggle_NosBench_Group1.ipynb                # NOSBENCH group 1 (101 problems)
├── MPECSS_Kaggle_NosBench_Group2.ipynb                # NOSBENCH group 2 (101 problems)
├── MPECSS_Kaggle_NosBench_Group3.ipynb                # NOSBENCH group 3 (101 problems)
├── MPECSS_Kaggle_NosBench_Group4.ipynb                # NOSBENCH group 4 (100 problems)
├── MPECSS_Kaggle_NosBench_Group5.ipynb                # NOSBENCH group 5 (100 problems)
├── MPECSS_Kaggle_NosBench_Group6.ipynb                # NOSBENCH group 6 (100 problems)
│
├── resumable_benchmark.py                              # Benchmark runner with resume/retry support
├── study_runner.py                                     # Study notebook helper (builds CLI commands)
├── __init__.py                                         # Python package marker
│
└── nosbench_splits/                                    # Problem lists for NOSBENCH groups
    ├── nosbench_group1_problems.txt                    # 101 problem filenames
    ├── nosbench_group2_problems.txt                    # 101 problem filenames
    ├── nosbench_group3_problems.txt                    # 101 problem filenames
    ├── nosbench_group4_problems.txt                    # 100 problem filenames
    ├── nosbench_group5_problems.txt                    # 100 problem filenames
    └── nosbench_group6_problems.txt                    # 100 problem filenames
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Notebook fails at `pip install` | Ensure Internet is enabled in Settings |
| "Benchmark path not found" | Verify the dataset is attached in the Input sidebar; check that the folder structure matches the expected layout |
| Notebook times out (12h limit) | This is a Kaggle limitation. For NOSBENCH, use the group-specific notebooks. For MacMPEC, results should complete within 12 hours. If not, use resume+retry |
| "ModuleNotFoundError: mpecss" | The `git clone` or `pip install` cell may have failed. Re-run the notebook |
| Out of memory (OOM) | Reduce `--workers` from 4 to 2 or 1. Each worker runs one problem at a time |
