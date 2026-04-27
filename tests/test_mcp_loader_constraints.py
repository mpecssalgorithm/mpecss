import json

import casadi as ca

from mpecss.helpers.loaders.macmpec_loader import load_macmpec


def _write_tiny_macmpec(tmp_path, *, ub_h=None):
    x = ca.SX.sym("x", 1)
    f_fun = ca.Function("f_tiny", [x], [x[0] ** 2])
    g_fun = ca.Function("g_empty", [x], [ca.SX(0, 1)])
    G_fun = ca.Function("G_tiny", [x], [ca.vertcat(x[0])])
    H_fun = ca.Function("H_tiny", [x], [ca.vertcat(x[0])])

    data = {
        "lbw": [-10.0],
        "ubw": [10.0],
        "w0": [1.0],
        "f_fun": f_fun.serialize(),
        "G_fun": G_fun.serialize(),
        "H_fun": H_fun.serialize(),
        "g_fun": g_fun.serialize(),
        "lbg": [],
        "ubg": [],
        "lbG": [-float("inf")],
        "lbH": [0.0],
    }
    if ub_h is not None:
        data["ubH"] = [ub_h]

    path = tmp_path / "tiny.nl.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_macmpec_loader_enforces_lower_only_free_g_sign(tmp_path):
    problem = load_macmpec(str(_write_tiny_macmpec(tmp_path)))
    info = problem["build_casadi"](0.0, 0.0)

    assert problem["G_is_free"] == [True]
    assert info["n_bounded_G"] == 1
    assert info["_bounded_G_idx"] == [0]


def test_macmpec_loader_keeps_box_mcp_free_g_globally_unconstrained(tmp_path):
    problem = load_macmpec(str(_write_tiny_macmpec(tmp_path, ub_h=10.0)))
    info = problem["build_casadi"](0.0, 0.0)

    assert problem["G_is_free"] == [True]
    assert info["n_bounded_G"] == 0
    assert info["_bounded_G_idx"] == []
    assert info["n_ubH"] == 1
