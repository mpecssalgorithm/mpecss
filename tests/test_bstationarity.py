import numpy as np
import casadi as ca

from mpecss.phase_3.bstationarity import certify_bstationarity, clear_jacobian_cache


def _infeasible_lower_only_free_g_problem():
    x = ca.SX.sym("x", 1)
    G_ca = ca.Function("G_test_infeasible", [x], [ca.vertcat(-1.0)])
    H_ca = ca.Function("H_test_infeasible", [x], [ca.vertcat(x[0])])

    def build_casadi(_t=0.0, _delta=0.0, smoothing="product"):
        del smoothing
        y = ca.SX.sym("x", 1)
        return {
            "x": y,
            "f": y[0] ** 2,
            "g": ca.SX(0, 1),
            "lbg": [],
            "ubg": [],
            "lbx": [-10.0],
            "ubx": [10.0],
        }

    return {
        "name": "test_infeasible_lower_only_free_g",
        "family": "unit",
        "n_x": 1,
        "n_comp": 1,
        "n_con": 0,
        "G_fn": G_ca,
        "H_fn": H_ca,
        "build_casadi": build_casadi,
        "lbG_eff": [-np.inf],
        "lbH_eff": [0.0],
        "G_is_free": [True],
    }


def test_bstationarity_does_not_certify_sign_infeasible_mcp_point():
    clear_jacobian_cache()
    problem = _infeasible_lower_only_free_g_problem()

    is_bstat, lpec_obj, licq_holds, details = certify_bstationarity(
        np.array([0.0]),
        problem,
        tol=1e-8,
    )

    assert is_bstat is False
    assert lpec_obj == float("inf")
    assert licq_holds is False
    assert details["lpec_status"] == "infeasible_skip"
    assert details["mcp_feas_res"] > 0.0
