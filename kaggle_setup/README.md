# MPECSS Kaggle Benchmark Setup

This folder contains everything needed to run the benchmarks on Kaggle.

# Quick Start Guide

## Step 1: Open Kaggle

Go to https://www.kaggle.com and sign in (or create a free account).

## Step 2: Create a New Notebook

1. Click **"Create"** → **"New Notebook"**
2. Or go to: https://www.kaggle.com/notebooks

## Step 3: Add the Dataset

1. In the right sidebar, click **Input**
2. Click **+ Add Input** and attach a dataset that contains the benchmark JSON files
3. The dataset name can be anything; notebooks auto-detect paths under `/kaggle/input/*/benchmarks/...`

Required benchmark folder layout:
- `benchmarks/mpeclib/mpeclib-json`
- `benchmarks/macmpec/macmpec-json`
- `benchmarks/nosbench/nosbench-json`

You can use the official Kaggle dataset ([mpecss-benchmarks](https://www.kaggle.com/datasets/mpecssalgorithm/mpecss-benchmarks)) or upload the same folder structure from your own dataset.

## Step 4: Upload a Notebook

Choose the notebook for your benchmark:

| Benchmark | Notebook to Upload |
|-----------|-------------------|
| MPECLib (92 problems) | `MPECSS_Kaggle_MPECLib.ipynb` |
| MacMPEC (191 problems) | `MPECSS_Kaggle_MacMPEC.ipynb` |
| MacMPEC ablation (No Phase-I) | `MPECSS_Kaggle_MacMPEC_Ablation_NoPhaseI.ipynb` |
| MacMPEC ablation (Fixed t-update in Phase-II) | `MPECSS_Kaggle_MacMPEC_Ablation_FixedPhaseII.ipynb` |
| MacMPEC seed robustness (Seed 11 only) | `MPECSS_Kaggle_MacMPEC_SeedRobustness_Seed11.ipynb` |
| MacMPEC seed robustness (Seed 123 only) | `MPECSS_Kaggle_MacMPEC_SeedRobustness_Seed123.ipynb` |
| MacMPEC parameter sensitivity (`t0=0.1`) | `MPECSS_Kaggle_MacMPEC_ParamSensitivity_t0_0p1.ipynb` |
| MacMPEC parameter sensitivity (`t0=10.0`) | `MPECSS_Kaggle_MacMPEC_ParamSensitivity_t0_10.ipynb` |
| MacMPEC parameter sensitivity (`kappa=0.3`) | `MPECSS_Kaggle_MacMPEC_ParamSensitivity_kappa_0p3.ipynb` |
| MacMPEC parameter sensitivity (`kappa=0.8`) | `MPECSS_Kaggle_MacMPEC_ParamSensitivity_kappa_0p8.ipynb` |
| NosBench Group 1 (101 problems) | `MPECSS_Kaggle_NosBench_Group1.ipynb` |
| NosBench Group 2 (101 problems) | `MPECSS_Kaggle_NosBench_Group2.ipynb` |
| NosBench Group 3 (101 problems) | `MPECSS_Kaggle_NosBench_Group3.ipynb` |
| NosBench Group 4 (100 problems) | `MPECSS_Kaggle_NosBench_Group4.ipynb` |
| NosBench Group 5 (100 problems) | `MPECSS_Kaggle_NosBench_Group5.ipynb` |
| NosBench Group 6 (100 problems) | `MPECSS_Kaggle_NosBench_Group6.ipynb` |

**To upload:**
1. Click **"File"** → **"Import Notebook"**
2. Select the `.ipynb` file from this folder

## Step 5: Enable Internet

1. Click **"Settings"**
2. Turn **Internet** on if the notebook needs to install packages or fetch assets

## Step 6: Run the Notebook

1. Click on the **"Save Version"** button in the top right corner of the notebook
2. Write any name for the version (for example `MPECLib_Benchmark_Run`)
3. Keep the Version type as **"Save & Run All (commit)"**
4. Click **"Save"** to start the run

This will trigger the execution of the notebook and save the outputs to Kaggle's output directory.


## Step 7: Download Results

After completion:
1. Go to **Output** tab
2. Download `/kaggle/working/outputs.zip` for a single archive
3. You can also inspect the unpacked files under `/kaggle/working/outputs/`

---


**Note:** NosBench is split into 6 groups (101/101/101/100/100/100 problems) to stay within Kaggle's 12-hour runtime limit. After collecting all 6 CSV outputs, merge them for the final NosBench table.

## Folder Structure

```
kaggle_setup/
├── README.md                           # This file
|
├── MPECSS_Kaggle_MPECLib.ipynb        # MPECLib benchmark notebook
├── MPECSS_Kaggle_MacMPEC.ipynb        # MacMPEC benchmark notebook
├── MPECSS_Kaggle_MacMPEC_Ablation_NoPhaseI.ipynb      # MacMPEC ablation without Phase-I
├── MPECSS_Kaggle_MacMPEC_Ablation_FixedPhaseII.ipynb  # MacMPEC ablation with fixed Phase-II update
├── MPECSS_Kaggle_MacMPEC_SeedRobustness_Seed11.ipynb   # MacMPEC seed-11 notebook
├── MPECSS_Kaggle_MacMPEC_SeedRobustness_Seed123.ipynb  # MacMPEC seed-123 notebook
├── MPECSS_Kaggle_MacMPEC_ParamSensitivity_t0_0p1.ipynb # MacMPEC t0=0.1 notebook
├── MPECSS_Kaggle_MacMPEC_ParamSensitivity_t0_10.ipynb  # MacMPEC t0=10.0 notebook
├── MPECSS_Kaggle_MacMPEC_ParamSensitivity_kappa_0p3.ipynb # MacMPEC kappa=0.3 notebook
├── MPECSS_Kaggle_MacMPEC_ParamSensitivity_kappa_0p8.ipynb # MacMPEC kappa=0.8 notebook
├── MPECSS_Kaggle_NosBench_Group1.ipynb # NosBench Group 1 notebook
├── MPECSS_Kaggle_NosBench_Group2.ipynb # NosBench Group 2 notebook
├── MPECSS_Kaggle_NosBench_Group3.ipynb # NosBench Group 3 notebook
├── MPECSS_Kaggle_NosBench_Group4.ipynb # NosBench Group 4 notebook
├── MPECSS_Kaggle_NosBench_Group5.ipynb # NosBench Group 5 notebook
├── MPECSS_Kaggle_NosBench_Group6.ipynb # NosBench Group 6 notebook
│
├── resumable_benchmark.py              # Benchmark runner with resume support
├── study_runner.py                     # Shared helper for MacMPEC study notebooks
│
└── nosbench_splits/                    # Problem lists for NosBench groups
    ├── nosbench_group1_problems.txt
    ├── nosbench_group2_problems.txt
    ├── nosbench_group3_problems.txt
    ├── nosbench_group4_problems.txt
    ├── nosbench_group5_problems.txt
    └── nosbench_group6_problems.txt

```
