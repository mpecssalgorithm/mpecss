from typing import List

import numpy as np

from mpecss.helpers.loaders.macmpec_loader import evaluate_GH


def identify_active_set(z, problem, tol=1e-06):
    # Step 1: "Parting the Sea" (Active Set Identification).
    G, H = evaluate_GH(z, problem)
    n_comp = problem['n_comp']
    ubH_finite = problem.get('ubH_finite', [])   # [(idx, ub_val), ...]
    ubH_map = {i: ub for i, ub in ubH_finite}

    lbG_eff = np.array(problem.get('lbG_eff', np.zeros(len(G))), dtype=float)
    lbH_eff = np.array(problem.get('lbH_eff', np.zeros(len(H))), dtype=float)
    G_is_free = np.array(problem.get('G_is_free', [False] * len(G)), dtype=bool)
    H_is_free = np.array(problem.get('H_is_free', [False] * len(H)), dtype=bool)
    G_shifted = np.array(G, dtype=float)
    H_shifted = np.array(H, dtype=float)
    G_shifted[~G_is_free] = G_shifted[~G_is_free] - lbG_eff[~G_is_free]
    H_shifted[~H_is_free] = H_shifted[~H_is_free] - lbH_eff[~H_is_free]

    I1: List[int] = []
    I2: List[int] = []
    I3: List[int] = []
    I_biactive: List[int] = []

    for i in range(n_comp):
        g_val = abs(float(G_shifted[i]))
        h_val = abs(float(H_shifted[i]))

        if i in ubH_map:
            slack_upper = abs(float(H[i]) - ubH_map[i])
            if slack_upper < tol:
                I3.append(i)
                continue

        if g_val < tol and h_val < tol:
            I_biactive.append(i)
            if g_val <= h_val:
                I1.append(i)
            else:
                I2.append(i)
        elif g_val < tol:
            I1.append(i)
        elif h_val < tol:
            I2.append(i)
        else:
            if g_val <= h_val:
                I1.append(i)
            else:
                I2.append(i)

    return I1, I2, I_biactive, I3
