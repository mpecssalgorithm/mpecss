from typing import Any, Dict


_OBJ_ACCEPT_REL = 1e-10
_OBJ_ACCEPT_ABS = 1e-10


def objective_not_worse(f_new: float, f_ref: float) -> bool:
    tol = max(abs(f_ref) * _OBJ_ACCEPT_REL, _OBJ_ACCEPT_ABS)
    return f_new <= f_ref + tol


def invalidate_stationarity_claim(results: Dict[str, Any], reason: str) -> None:
    # A polished point is a new iterate; old stationarity certificates no longer apply.
    results['status'] = 'stationarity_unverifiable'
    results['stationarity'] = 'FAIL'
    results['sign_test_pass'] = None
    results['b_stationarity'] = None
    results['lpec_obj'] = None
    results['licq_holds'] = None
    results['bstat_details'] = {
        'lpec_status': 'invalidated_by_bnlp',
        'classification': 'uncertified',
        'reason': reason,
    }
