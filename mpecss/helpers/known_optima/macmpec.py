# The "Answer Key" (MacMPEC): Checking our results.

KNOWN_OPTIMA = {}   # populated at runtime from bytecode data

RED_FLAG_PROBLEMS = set()  # problems with known convergence issues

DEFAULT_REFERENCE_FILENAME = 'reference_optima.json'


def get_known_optimum(problem_name):
    # Return the known optimal value for a problem, or None.
    if not problem_name:
        return None
    return KNOWN_OPTIMA.get(str(problem_name))


def compute_optimality_gap(f_final, problem_name, tol=1e-2, better_tol=None):
    # Compute relative optimality gap with literature-justified tolerance.
    f_star = get_known_optimum(problem_name)
    if f_star is None:
        return (None, None)
    if better_tol is None:
        better_tol = 5.0 * tol
    denom = max(1.0, abs(float(f_star)))
    gap = abs(float(f_final) - float(f_star)) / denom
    ok = gap < (better_tol if float(f_final) <= float(f_star) else tol)
    return (gap, ok)


def load_reference(filepath=None):
    # Load reference optima from a JSON file.
    import json
    import os

    target = get_reference_path(filepath)
    if target is None or not os.path.isfile(target):
        return 0
    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    KNOWN_OPTIMA.clear()
    for k, v in data.items():
        try:
            KNOWN_OPTIMA[str(k)] = float(v)
        except Exception:
            continue
    return len(KNOWN_OPTIMA)


def get_reference_path(filepath=None):
    # Return the resolved path to the reference optima file.
    import os

    if filepath:
        return filepath
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(repo_root, DEFAULT_REFERENCE_FILENAME)


def get_known_optimum_nosbench(problem_name):
    # Return the known optimal value for a NOSBENCH problem, or None.
    return get_known_optimum(problem_name)


def compute_optimality_gap_nosbench(f_final, problem_name, tol=1e-2, better_tol=None):
    # Compute optimality gap for a NOSBENCH problem.
    return compute_optimality_gap(f_final, problem_name, tol=tol, better_tol=better_tol)


def set_reference_dict(reference_dict):
    # Override the in-memory reference optima dict.
    KNOWN_OPTIMA.clear()
    for k, v in (reference_dict or {}).items():
        try:
            KNOWN_OPTIMA[str(k)] = float(v)
        except Exception:
            continue
