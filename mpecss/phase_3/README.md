# `mpecss/phase_3/` — Phase III: Polishing & B-Stationarity Certification

This subpackage implements Phase III of the MPECSS algorithm: solution polishing and stationarity certification.

## Modules

| Module | Description |
|---|---|
| `bnlp_polish.py` | `bnlp_polish()` — Branch-NLP polishing with active-set identification |
| `bnlp_polish_sets.py` | `identify_active_set()` — partitions complementarity indices into I1/I2/biactive/I3 |
| `bnlp_polish_utils.py` | Utility functions for BNLP polishing |
| `bstationarity.py` | `certify_bstationarity()` — B-stationarity certification via LPEC enumeration; `check_mpec_licq()` — LICQ check |
| `lpec_refine.py` | `lpec_refinement_loop()` — optional LPEC-guided iterative refinement |

## How It Works

### Step 1: BNLP Polish

1. Identify the active set at the converged solution:
   - `I1` (G-active): G_i ≈ 0, H_i > 0
   - `I2` (H-active): H_i ≈ 0, G_i > 0
   - `I_biactive`: Both G_i ≈ 0 and H_i ≈ 0
   - `I3`: Neither active
2. Solve a "branch NLP" with complementarity constraints fixed according to the partition
3. Try alternative partitions of biactive indices to find the best branch

### Step 2: LICQ Check

Check if the MPEC-LICQ (Linear Independence Constraint Qualification) holds at the solution by evaluating the rank of the active constraint Jacobian.

**Shortcut**: If LICQ holds and the Phase II sign test passed, then S-stationarity ⟺ B-stationarity. The solution can be certified as B-stationary without LPEC enumeration.

### Step 3: LPEC Enumeration (if needed)

If LICQ does not hold or the sign test failed:
1. Enumerate all possible biactive partitions
2. For each partition, solve a Linear Program with Equilibrium Constraints (LPEC)
3. If no descent direction is found across all partitions → B-stationary
4. If a descent direction is found → C-stationary (not B-stationary)

## Classification Output

| Result | Meaning |
|---|---|
| `B-stationary (S + LICQ)` | S-stationary under LICQ → B-stationary by equivalence |
| `B-stationary (LPEC)` | LPEC enumeration found no descent direction |
| `C-stationary` | Complementarity satisfied but not B-stationary |
| `FAIL` | Certification could not be completed |

See the parent [`mpecss/README.md`](../README.md) for full documentation.
