# MacMPEC Benchmark Dataset

MacMPEC is a benchmark suite of 191 Mathematical Programs with Equilibrium Constraints (MPECs) curated by Sven Leyffer at Argonne National Laboratory. It is one of the most widely used MPEC benchmark collections in the optimization literature.

---

## What Is MacMPEC?

MacMPEC was created to provide a comprehensive, diverse set of MPEC test problems for evaluating solvers. The problems come from various application domains:

- **Bilevel optimization**: Leader-follower games, Stackelberg equilibria
- **Economic equilibrium**: Market models, pricing problems
- **Engineering design**: Structural optimization with contact constraints
- **Game theory**: Nash equilibrium computation, multi-leader-follower games
- **Network design**: Traffic equilibrium, facility location

The original problems are defined in **AMPL format** (`.mod`/`.dat` files). For MPECSS, they have been converted to **CasADi JSON format** (`.nl.json`) by Anton Pozharskiy (University of Freiburg / SYSCOP group).

**Original source**: [MacMPEC wiki](https://wiki.mcs.anl.gov/leyffer/index.php/MacMPEC)

---

## What Is Included Here

| File/Folder | Description |
|---|---|
| `README.md` | This file |
| `macmpec-json/` | **191 CasADi JSON files** — the primary benchmark data (excluded via `.gitignore`; download separately) |

> **The JSON files are not included in the Git repository** due to their size. See the Download section below.

---

## Dataset Metadata

| Property | Value |
|---|---|
| **Total problems** | 191 |
| **Primary format** | CasADi JSON (`.nl.json`) |
| **Original format** | AMPL model/data files (`.mod`/`.dat`) |
| **Problem sizes** | 2 to ~1000+ variables; 1 to ~500+ complementarity pairs |
| **Expected folder name** | `macmpec-json/` |

---

## Download & Setup

### Step 1: Download the JSON files

| Format | Source |
|---|---|
| CasADi JSON (`macmpec-json/`) | [SYSCOP Cloud](https://cloud.syscop.de/s/rBnTMocFoLcNLWG) |
| Original AMPL (`.mod`/`.dat`) | [MacMPEC wiki](https://wiki.mcs.anl.gov/leyffer/index.php/MacMPEC) |

Alternatively, use the [Kaggle dataset](https://www.kaggle.com/datasets/mpecssalgorithm/mpecss-benchmarks) which includes all three benchmark suites.

### Step 2: Place the files

Download and extract the JSON files into the `macmpec-json/` subdirectory:

```text
macmpec/
├── README.md
└── macmpec-json/            ← Download and place 191 .nl.json files here
    ├── bard1.nl.json
    ├── bilevel1.nl.json
    ├── ...
    └── tollmpec.nl.json
```

### Verification

After downloading, you should have exactly 191 JSON files:

```bash
ls benchmarks/macmpec/macmpec-json/*.nl.json | wc -l
# Expected output: 191
```

---

## Usage in MPECSS

### Programmatic usage

```python
from mpecss.helpers.loaders import load_macmpec
from mpecss import run_mpecss

# Load a single problem
problem = load_macmpec("benchmarks/macmpec/macmpec-json/bard1.nl.json")
print(f"Name: {problem['name']}, n_x={problem['n_x']}, n_comp={problem['n_comp']}")

# Solve it
z0 = problem["x0_fn"](seed=42)
result = run_mpecss(problem, z0, params={"seed": 42})
print(f"Status: {result['status']}, Stationarity: {result['stationarity']}")
```

### Benchmark usage

```bash
python kaggle_setup/resumable_benchmark.py --dataset macmpec --workers 4 --timeout 3600 --seed 42
```

### Ablation, seed, and parameter studies

MacMPEC is used as the primary suite for ablation and sensitivity studies. See `kaggle_setup/README.md` for the full list of study notebooks.

---

## Ownership and Attribution

All original rights to MacMPEC belong to the original authors and rights holders.
This folder is provided as benchmark metadata and source-reference material only.
No original problem data is redistributed here.

## Credits

- **Creator**: Sven Leyffer (Argonne National Laboratory)
- **JSON Conversion**: Anton Pozharskiy (University of Freiburg / SYSCOP group)

---

## Citing MacMPEC

If you use MacMPEC in your research, please cite the original source:

```
S. Leyffer. MacMPEC — AMPL collection of MPECs.
https://wiki.mcs.anl.gov/leyffer/index.php/MacMPEC
```
