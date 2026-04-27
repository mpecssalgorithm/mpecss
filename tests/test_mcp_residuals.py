import numpy as np

from mpecss.helpers.comp_residuals import (
    benchmark_feas_res,
    complementarity_residual,
    homotopy_comp_res,
)


def _constant_problem(g_value, h_value, **metadata):
    return {
        "G_fn": lambda _x: np.array([g_value], dtype=float),
        "H_fn": lambda _x: np.array([h_value], dtype=float),
        "n_comp": 1,
        **metadata,
    }


def test_homotopy_residual_uses_shifted_lower_bounds():
    problem = _constant_problem(
        2.0,
        1.0,
        lbG_eff=[0.0],
        lbH_eff=[1.0],
        G_is_free=[False],
    )

    assert homotopy_comp_res(np.zeros(1), problem) == 0.0
    assert benchmark_feas_res(np.zeros(1), problem) == 0.0


def test_lower_only_free_g_requires_mcp_sign():
    problem = _constant_problem(
        -1.0,
        0.0,
        lbG_eff=[-np.inf],
        lbH_eff=[0.0],
        G_is_free=[True],
    )

    assert complementarity_residual(np.zeros(1), problem) == 1.0
    assert benchmark_feas_res(np.zeros(1), problem) == 1.0


def test_box_mcp_free_g_branch_semantics():
    base = {
        "lbG_eff": [-np.inf],
        "lbH_eff": [0.0],
        "G_is_free": [True],
        "ubH_finite": [(0, 10.0)],
    }

    valid_lower = _constant_problem(2.0, 0.0, **base)
    invalid_lower = _constant_problem(-2.0, 0.0, **base)
    valid_upper = _constant_problem(-2.0, 10.0, **base)
    invalid_upper = _constant_problem(2.0, 10.0, **base)
    valid_interior = _constant_problem(0.0, 5.0, **base)
    invalid_interior_pos = _constant_problem(2.0, 5.0, **base)
    invalid_interior_neg = _constant_problem(-2.0, 5.0, **base)

    assert benchmark_feas_res(np.zeros(1), valid_lower) == 0.0
    assert benchmark_feas_res(np.zeros(1), invalid_lower) > 0.0
    assert benchmark_feas_res(np.zeros(1), valid_upper) == 0.0
    assert benchmark_feas_res(np.zeros(1), invalid_upper) > 0.0
    assert benchmark_feas_res(np.zeros(1), valid_interior) == 0.0
    assert benchmark_feas_res(np.zeros(1), invalid_interior_pos) > 0.0
    assert benchmark_feas_res(np.zeros(1), invalid_interior_neg) > 0.0
