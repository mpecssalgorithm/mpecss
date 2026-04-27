DEFAULT_IPOPT_OPTS = {
    'tol': 1e-8,
    'acceptable_tol': 1e-6,
    'print_level': 0,
    'sb': 'yes',  # Suppress IPOPT banner (copyright/license messages)
    'max_iter': 3000,
    'mu_strategy': 'adaptive',
    'mu_oracle': 'quality-function',
    'linear_solver': 'mumps',
    'mumps_mem_percent': 500,
    'mumps_pivtol': 1e-6,
    'mumps_scaling': 77,
    'mumps_pivot_order': 5,
    'nlp_scaling_method': 'gradient-based',
    'max_soc': 4,
    'watchdog_trial_iter_max': 3,
}

_WARM_START_EXTRA_OPTS = {
    'warm_start_init_point': 'yes',
    'warm_start_bound_push': 1e-09,
    'warm_start_bound_frac': 1e-09,
    'warm_start_slack_bound_frac': 1e-09,
    'warm_start_slack_bound_push': 1e-09,
    'warm_start_mult_bound_push': 1e-09
}

_FALLBACK_TRIGGER_STATUSES = {
    'Invalid_Option',
    'Restoration_Failed',
    'Invalid_Number_Detected',
    'Error_In_Step_Computation',
    'Infeasible_Problem_Detected',
    'Maximum_Iterations_Exceeded',
    'Diverging_Iterates',          # ex9.1.x, design-cent-4: IPOPT diverges
}

_SOLVER_FALLBACK_CHAIN = [
    {'mehrotra_algorithm': 'yes', 'mu_oracle': 'probing'},
    {'tol': 1e-08, 'acceptable_tol': 1e-06},
    {
        'nlp_scaling_method': 'gradient-based',
        'bound_relax_factor': 1e-8,
        'tol': 1e-06,
        'acceptable_tol': 1e-04,
    },
]
