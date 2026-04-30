# `mpecss/helpers/loaders/` — Problem Instance Loaders

This subpackage provides loader functions for reading benchmark problem instances from JSON files.

## Available Loaders

| Function | Module | Input Format | Benchmark Suite |
|---|---|---|---|
| `load_mpeclib(path)` | `mpeclib_loader.py` | `.nl.json` | MPECLib (92 problems from GAMS World) |
| `load_macmpec(path)` | `macmpec_loader.py` | `.nl.json` | MacMPEC (191 problems from Sven Leyffer) |
| `load_nosbench(path)` | `nosbench_loader.py` | `.json` | NOSBENCH (603 problems from nosnoc) |

## Usage

```python
from mpecss.helpers.loaders import load_mpeclib, load_macmpec, load_nosbench

# Load a problem
problem = load_mpeclib("benchmarks/mpeclib/mpeclib-json/bard1.nl.json")

# Inspect problem dimensions
print(f"Variables: {problem['n_x']}")
print(f"Complementarity pairs: {problem['n_comp']}")
print(f"General constraints: {problem['n_con']}")
```

## Output: ProblemSpec Dictionary

All loaders return a dictionary conforming to `mpecss.contracts.ProblemSpec`:

| Key | Type | Description |
|---|---|---|
| `name` | `str` | Problem name (derived from filename) |
| `n_x` | `int` | Total number of decision variables |
| `n_comp` | `int` | Number of complementarity pairs |
| `n_con` | `int` | Number of general (non-complementarity) constraints |
| `n_p` | `int` | Number of parameters |
| `family` | `str` | Problem family or category |
| `x0_fn` | `Callable[[int], ndarray]` | Function to generate initial point from seed |
| `f_fn` | `Callable[[ndarray], float]` | Objective function evaluator |
| `G_fn` | `Callable[[ndarray], ndarray]` | G-side complementarity function |
| `H_fn` | `Callable[[ndarray], ndarray]` | H-side complementarity function |
| `build_casadi` | `Callable` | Builds the CasADi NLP for a given (t, delta) pair |
| `lbx`, `ubx` | `List[float]` | Variable bounds |
| `lbg`, `ubg` | `List[float]` | Constraint bounds |
| `unsupported_model_reason` | `Optional[str]` | If set, the model is not fully supported |

See the parent [`mpecss/README.md`](../../README.md) for full documentation.
