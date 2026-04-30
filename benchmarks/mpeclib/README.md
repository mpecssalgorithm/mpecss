# MPECLib Benchmark Dataset

MPECLib is a collection of 92 Mathematical Programs with Equilibrium Constraints (MPECs) from the GAMS World library. These are classic, well-studied MPEC test problems covering a range of application domains including bilevel programming, contact problems, and economic equilibrium.

---

## What Is MPECLib?

MPECLib was originally created as part of the [GAMS World](https://www.gams.com/latest/gamslib_ml/libhtml/) project to provide a standardized set of MPEC test problems for solver comparison. Each problem is defined in the GAMS Scalar (`.gms`) format, a modeling language widely used in mathematical programming.

For MPECSS, the problems have been converted to **CasADi JSON format** (`.nl.json`), which encodes the problem as a serialized CasADi computational graph. This allows the solver to use CasADi's automatic differentiation capabilities.

**Original source**: [GAMS World MPECLib repository](https://github.com/GAMS-dev/gamsworld/tree/master/MPECLib)

---

## What Is Included Here

| File/Folder | Description |
|---|---|
| `README.md` | This file |
| `convert_mpeclib.py` | Python script to convert GMS files → CasADi JSON format |
| `mpeclib-json/` | **92 CasADi JSON files** — the primary benchmark data (included in repository) |
| `mpeclib-gms/` | Original GAMS source files (excluded via `.gitignore`; download separately if needed) |

---

## Dataset Metadata

| Property | Value |
|---|---|
| **Total problems** | 92 |
| **Primary format** | CasADi JSON (`.nl.json`) |
| **Original format** | GAMS scalar files (`.gms`) |
| **Problem sizes** | 2 to ~600 variables; 1 to ~200 complementarity pairs |
| **Expected JSON folder** | `mpeclib-json/` |
| **Expected GMS folder** | `mpeclib-gms/` (optional, only needed for re-conversion) |

### Example Problems

| Problem | Variables | Comp. Pairs | Description |
|---|---|---|---|
| `bard1.nl.json` | 5 | 3 | Bard's bilevel example #1 |
| `ralph1.nl.json` | 2 | 1 | Ralph's simple MPEC |
| `scholtes4.nl.json` | 3 | 1 | Scholtes's regularization test case |
| `outrata31.nl.json` | ~30 | ~10 | Outrata's contact mechanics problem |

---

## Download & Setup

### The JSON files are already included

The `mpeclib-json/` directory is **included in the Git repository** — no additional download is needed:

```bash
git clone https://github.com/mpecssalgorithm/mpecss.git
ls mpecss/benchmarks/mpeclib/mpeclib-json/
# → bard1.nl.json, bard3.nl.json, ..., (92 files)
```

### Expected folder structure

```text
mpeclib/
├── README.md
├── convert_mpeclib.py
├── mpeclib-json/            ← 92 .nl.json files (included in repo)
│   ├── bard1.nl.json
│   ├── bard3.nl.json
│   ├── ...
│   └── scholtes4.nl.json
└── mpeclib-gms/             ← 92 .gms files (optional, download separately)
    ├── bard1.gms
    ├── ...
    └── scholtes4.gms
```

### Downloading the original GMS files (optional)

If you want to inspect or re-convert the original GAMS source:

| Format | Source |
|---|---|
| CasADi JSON (`mpeclib-json/`) | Already included; also on [MPECSS GitHub](https://github.com/mpecssalgorithm/mpecss/tree/main/benchmarks/mpeclib/mpeclib-json) |
| Original GMS (`mpeclib-gms/`) | [GAMS World GitHub](https://github.com/GAMS-dev/gamsworld/tree/master/MPECLib) |

---

## Re-converting from GMS to JSON

If you need to regenerate the JSON files from the GAMS source:

1. Download the `.gms` files from [GAMS World](https://github.com/GAMS-dev/gamsworld/tree/master/MPECLib) and place them in `mpeclib-gms/`
2. Run the conversion script:

   ```bash
   python convert_mpeclib.py
   ```

3. The script will parse each `.gms` file, build the CasADi symbolic representation, and write the corresponding `.nl.json` file to `mpeclib-json/`.

> **Note**: The conversion script uses AI-generated code that has been reviewed and modified by the authors. The authors take full responsibility for the conversion.

---

## Usage in MPECSS

### Programmatic usage

```python
from mpecss.helpers.loaders import load_mpeclib
from mpecss import run_mpecss

# Load a single problem
problem = load_mpeclib("benchmarks/mpeclib/mpeclib-json/bard1.nl.json")
print(f"Name: {problem['name']}, n_x={problem['n_x']}, n_comp={problem['n_comp']}")

# Solve it
z0 = problem["x0_fn"](seed=42)
result = run_mpecss(problem, z0, params={"seed": 42})
print(f"Status: {result['status']}, Stationarity: {result['stationarity']}")
```

### Benchmark usage

```bash
python kaggle_setup/resumable_benchmark.py --dataset mpeclib --workers 4 --timeout 3600 --seed 42
```

---

## Ownership and Attribution

All original rights to MPECLib belong to the original authors and rights holders.
This folder is provided as benchmark metadata and source-reference material only.

## Credits

- **Original Creator**: Steven Dirkse (GAMS Development Corp.)
- **Maintenance**: Michael Bussieck (GAMS Development Corp.)
- **JSON Conversion**: Performed using the `convert_mpeclib.py` script included in this directory
