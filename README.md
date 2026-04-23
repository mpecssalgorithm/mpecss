# MPECSS: Scholtes Regularization with Adaptive Paths for Targeting B-stationary Points in MPECs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

---

## What is MPECSS?

MPECSS is an algorithm for Mathematical Programs with Equilibrium Constraints (MPECs) or Mathematical Programs with Complementarity Constraints (MPCCs). These problems are hard to solve because they are nonconvex and contain complementarity constraints.

MPECSS uses a three-phase approach:
1. **Phase I**: It starts by finding a feasible point that satisfies the constraints,
2. **Phase II**: It then iteratively refines this point to find a stationary solution,
3. **Phase III**: Finally, it polishes the solution and attempts to certify whether the point is B-stationary.

---

## Official Benchmark Workflow (Recommended for Reproducibility and Comparison with Published Results)

Official benchmark runs are maintained through the Kaggle notebooks in `kaggle_setup/`.

### 1. Open Kaggle
Create a new notebook and upload the dataset in right side 'Input' section `benchmarks` collected from the [kaggle](https://www.kaggle.com/datasets/mpecssalgorithm/mpecss-benchmarks) platform or [Github](https://github.com/mpecssalgorithm/mpecss/tree/main/benchmarks).

### 2. Pick the notebook that matches your run

**Main benchmark notebooks**
- `kaggle_setup/MPECSS_Kaggle_MPECLib.ipynb`
- `kaggle_setup/MPECSS_Kaggle_MacMPEC.ipynb`
- `kaggle_setup/MPECSS_Kaggle_NosBench_Group1.ipynb`
- `kaggle_setup/MPECSS_Kaggle_NosBench_Group2.ipynb`
- `kaggle_setup/MPECSS_Kaggle_NosBench_Group3.ipynb`
- `kaggle_setup/MPECSS_Kaggle_NosBench_Group4.ipynb`
- `kaggle_setup/MPECSS_Kaggle_NosBench_Group5.ipynb`
- `kaggle_setup/MPECSS_Kaggle_NosBench_Group6.ipynb`

**MacMPEC ablation notebooks**
- `kaggle_setup/MPECSS_Kaggle_MacMPEC_Ablation_NoPhaseI.ipynb`
- `kaggle_setup/MPECSS_Kaggle_MacMPEC_Ablation_FixedPhaseII.ipynb`

**MacMPEC seed-robustness notebooks**
- `kaggle_setup/MPECSS_Kaggle_MacMPEC_SeedRobustness_Seed11.ipynb`
- `kaggle_setup/MPECSS_Kaggle_MacMPEC_SeedRobustness_Seed42.ipynb`
- `kaggle_setup/MPECSS_Kaggle_MacMPEC_SeedRobustness_Seed123.ipynb`

**MacMPEC parameter-sensitivity notebooks**
- `kaggle_setup/MPECSS_Kaggle_MacMPEC_ParamSensitivity_t0_0p1.ipynb`
- `kaggle_setup/MPECSS_Kaggle_MacMPEC_ParamSensitivity_t0_1.ipynb`
- `kaggle_setup/MPECSS_Kaggle_MacMPEC_ParamSensitivity_t0_10.ipynb`
- `kaggle_setup/MPECSS_Kaggle_MacMPEC_ParamSensitivity_kappa_0p3.ipynb`
- `kaggle_setup/MPECSS_Kaggle_MacMPEC_ParamSensitivity_kappa_0p5.ipynb`
- `kaggle_setup/MPECSS_Kaggle_MacMPEC_ParamSensitivity_kappa_0p8.ipynb`

### 3 Run and Save Outputs
Click on the `Save Version` button in the top right corner of the notebook (Put the name of the version as `MPECLib Benchmark Run`, `MacMPEC Benchmark Run`, or `NosBench Group X Benchmark Run` and Keep the Version type as `Save & Run All (commit)`). This will trigger the execution of the notebook and save the outputs to Kaggle's output directory.

The Kaggle-specific guide is in `kaggle_setup/README.md`.

---

## Running Benchmarks

The repository contains three benchmark suites:

- **MPECLib**: 92 problems
- **MacMPEC**: 191 problems
- **NosBench**: 603 problems (split across six Kaggle notebooks - 101 / 101 / 101 / 100 / 100 / 100)

The supported path is the Kaggle notebooks under `kaggle_setup/`. Each notebook clones the repository, installs the package in editable mode, runs `mpecss/helpers/preflight_checks.py`, and calls `kaggle_setup/resumable_benchmark.py` to write artifacts to `/kaggle/working/outputs` plus the version note JSON.

MacMPEC also includes three study notebooks for ablation, seed robustness, and parameter sensitivity experiments.

---

## Understanding the Solver Output

| Status | What it means |
| :--- | :--- |
| **B-stationary** | Strongest certificate reported by MPECSS; Phase III verified B-stationarity under the implemented assumptions. |
| **C-stationary** | Weaker stationarity outcome; useful diagnostically, but not a B-stationarity certificate. |
| **Stationarity unverifiable** | Complementarity may be small, but the implemented checks did not certify the stationarity claim. |
| **Timeout** | The run exceeded the configured time budget. |
| **Complementarity infeasibility** | MPECSS did not find a point satisfying complementarity to tolerance; this is not a proof that no feasible solution exists. |
| **NLP Solver Failure** | The underlying NLP solver failed or returned an unreliable status. |

---

## Project Structure

```
.
├── mpecss/                 # Solver package
│   ├── helpers/            # Loaders, solver wrappers, and utilities
│   ├── phase_1/            # Phase I: Feasibility and starting point
│   ├── phase_2/            # Phase II: Solver loop
│   └── phase_3/            # Phase III: Polishing and verification
├── benchmarks/             # MPECLib, MacMPEC, and NosBench JSON suites
├── kaggle_setup/           # Kaggle notebooks, runner, and helpers
├── docs/                   # Workflow diagrams and supporting docs
├── verification/           # Reference results for verification
├── requirements.txt        # Mirror of runtime dependencies
└── LICENSE                 # Apache 2.0 License
```
---
## Note:
The authors used AI-assisted code generation while developing this repository. The generated material has been reviewed and edited by the authors, who remain responsible for the implementation and results. Users are encouraged to review the code, open issues, or contribute improvements on [GitHub](https://github.com/mpecssalgorithm/mpecss).

**Need Help?**
- Open an issue on [GitHub](https://github.com/mpecssalgorithm/mpecss/issues)
- Email: `mpecssalgorithm@gmail.com`

---
[License: Apache 2.0](LICENSE)
