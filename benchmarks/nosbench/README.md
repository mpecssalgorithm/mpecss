# NOSBENCH Benchmark Dataset

NOSBENCH is a benchmark collection of 603 Mathematical Programs with Equilibrium Constraints (MPECs) arising from **nonsmooth optimal control** problems. It is the largest of the three benchmark suites used in the MPECSS evaluation.

---

## What Is NOSBENCH?

NOSBENCH was created by the [SYSCOP group](https://www.syscop.de/) at the University of Freiburg as part of the [nosnoc](https://github.com/nosnoc/nosnoc) project. The problems arise from discretizing nonsmooth optimal control problems using the FESD (Finite Elements with Switch Detection) method, which naturally produces MPEC/MPCC structures.

### Application Domains

- **Nonsmooth dynamics**: Systems with friction, impacts, and switching behavior
- **Optimal control**: Trajectory optimization for hybrid dynamical systems
- **Time-optimal control**: Minimum-time problems with contact mechanics
- **Multi-stage systems**: Problems with varying numbers of stages and discretization levels

### Problem Structure

NOSBENCH problems are organized by **difficulty level** (1–4), where higher levels correspond to finer discretizations and more complementarity pairs:

| Level | Characteristics | Typical Size |
|---|---|---|
| Level 1 | Coarse discretization, few complementarity pairs | 10–50 variables, 5–25 comp. pairs |
| Level 2 | Moderate discretization | 50–200 variables, 25–100 comp. pairs |
| Level 3 | Fine discretization | 200–2000 variables, 100–1000 comp. pairs |
| Level 4 | Very fine discretization, large-scale | 2000–10000+ variables, 1000–5000+ comp. pairs |

The upstream NOSBENCH project is available at [GitHub](https://github.com/nosnoc/nosbench).

---

## What Is Included Here

| File/Folder | Description |
|---|---|
| `README.md` | This file |
| `nosbench-json/` | **603 CasADi JSON files** — the primary benchmark data (excluded via `.gitignore`; download separately) |
| `nosbench-mat/` | MATLAB structured archive organized by level (excluded via `.gitignore`; optional) |

> **The problem files are not included in the Git repository** due to their size. See the Download section below.

---

## Dataset Metadata

| Property | Value |
|---|---|
| **Total problems** | 603 |
| **Primary format** | CasADi JSON (`.json`) |
| **Supplementary format** | MATLAB structured archive (`.mat`, organized by level) |
| **Problem sizes** | 10 to 10000+ variables; 5 to 5000+ complementarity pairs |
| **Expected JSON folder** | `nosbench-json/` |
| **Expected MATLAB folder** | `nosbench-mat/` (optional) |
| **Batch split for Kaggle runs** | 101 / 101 / 101 / 100 / 100 / 100 |

---

## Download & Setup

### Step 1: Download the JSON files

| Format | Source |
|---|---|
| CasADi JSON (`nosbench-json/`) | [nosnoc/nosbench on GitHub](https://github.com/nosnoc/nosbench) |
| MATLAB archive (`nosbench-mat/`) | [nosnoc/nosbench on GitHub](https://github.com/nosnoc/nosbench) |

Alternatively, use the [Kaggle dataset](https://www.kaggle.com/datasets/mpecssalgorithm/mpecss-benchmarks) which includes all three benchmark suites.

### Step 2: Place the files

Download and place the files so the structure looks like:

```text
nosbench/
├── README.md
├── nosbench-json/               ← Download and place 603 .json files here
│   ├── problem_001.json
│   ├── problem_002.json
│   ├── ...
│   └── problem_603.json
└── nosbench-mat/                ← Download and place here (optional)
    ├── generators/
    ├── level1/
    ├── level2/
    ├── level3/
    └── level4/
```

### Verification

After downloading, you should have exactly 603 JSON files:

```bash
ls benchmarks/nosbench/nosbench-json/*.json | wc -l
# Expected output: 603
```

---

## Kaggle Batch Splitting

Because Kaggle enforces a 12-hour maximum runtime per notebook session, the 603 NOSBENCH problems are split into 6 groups for parallel execution:

| Group | Problems | Notebook | Problem List File |
|---|---|---|---|
| Group 1 | 101 | `MPECSS_Kaggle_NosBench_Group1.ipynb` | `kaggle_setup/nosbench_splits/nosbench_group1_problems.txt` |
| Group 2 | 101 | `MPECSS_Kaggle_NosBench_Group2.ipynb` | `kaggle_setup/nosbench_splits/nosbench_group2_problems.txt` |
| Group 3 | 101 | `MPECSS_Kaggle_NosBench_Group3.ipynb` | `kaggle_setup/nosbench_splits/nosbench_group3_problems.txt` |
| Group 4 | 100 | `MPECSS_Kaggle_NosBench_Group4.ipynb` | `kaggle_setup/nosbench_splits/nosbench_group4_problems.txt` |
| Group 5 | 100 | `MPECSS_Kaggle_NosBench_Group5.ipynb` | `kaggle_setup/nosbench_splits/nosbench_group5_problems.txt` |
| Group 6 | 100 | `MPECSS_Kaggle_NosBench_Group6.ipynb` | `kaggle_setup/nosbench_splits/nosbench_group6_problems.txt` |

After all 6 groups complete, merge the CSV outputs:

```python
import pandas as pd
import glob

csvs = glob.glob("nosbench_full_*.csv")
df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
df.to_csv("nosbench_full_merged.csv", index=False)
print(f"Merged {len(df)} problems from {len(csvs)} groups")
```

---

## Usage in MPECSS

### Programmatic usage

```python
from mpecss.helpers.loaders import load_nosbench
from mpecss import run_mpecss

# Load a single problem
problem = load_nosbench("benchmarks/nosbench/nosbench-json/problem_001.json")
print(f"Name: {problem['name']}, n_x={problem['n_x']}, n_comp={problem['n_comp']}")

# Solve it
z0 = problem["x0_fn"](seed=42)
result = run_mpecss(problem, z0, params={"seed": 42})
print(f"Status: {result['status']}, Stationarity: {result['stationarity']}")
```

### Benchmark usage (all 603 problems locally)

```bash
python kaggle_setup/resumable_benchmark.py --dataset nosbench --workers 4 --timeout 3600 --seed 42
```

### Benchmark usage (single group)

```bash
python kaggle_setup/resumable_benchmark.py --dataset nosbench --workers 4 --timeout 3600 --seed 42 \
    --problem-list kaggle_setup/nosbench_splits/nosbench_group1_problems.txt
```

---

## Usage Notes

- After downloading, load instances from `benchmarks/nosbench/nosbench-json/`.
- Each `*.json` file represents one problem instance.
- For batch Kaggle runs, problems are split into 6 groups. Point each group's notebook at the `nosbench-json/` subfolder, not the parent.
- This dataset is solver-agnostic; use any compatible runner pipeline.

---

## Ownership and Attribution

All original rights to NOSBENCH belong to the original authors and rights holders.
This folder is provided as benchmark metadata and source-reference material only.
No original problem data is redistributed here.

## Credits

- **Creators**: Armin Nurkanović, Anton Pozharskiy, and Moritz Diehl (University of Freiburg / SYSCOP)

---

## Citing NOSBENCH

If you use this suite in your research, please cite:

```bibtex
@article{Nurkanovic2024,
  title = {Solving mathematical programs with complementarity constraints arising in nonsmooth optimal control},
  author = {Nurkanović, Armin and Pozharskiy, Anton and Diehl, Moritz},
  journal = {Vietnam Journal of Mathematics},
  year = {2024}
}
```
