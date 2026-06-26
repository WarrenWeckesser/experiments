methods = [
    "BFGS",
    "CG",
    "COBYLA",
    "COBYQA",
    "dogleg",
    "L-BFGS-B",
    "Nelder-Mead",
    "Newton-CG",
    "Powell",
    "SLSQP",
    "TNC",
    "trust-constr",
    "trust-exact",
    "trust-krylov",
    "trust-ncg",
]

# These methods do not use jac.  A warning is generated if jac is given.
no_jac = ["COBYLA", "COBYQA", "Nelder-Mead", "Powell"]

# These methods *require* jac.
require_jac = ["dogleg", "Newton-CG", "trust-ncg", "trust-krylov", "trust-exact"]

# These methods do not use hess; a warning is generated if hess is given.
no_hess = ["BFGS", "CG", "L-BFGS-B", "SLSQP", "TNC"]

# dogleg and trust-exact *require* hess.
# trust_krylov and trust-ncg *require* either hess or hessp.
require_hess = ["dogleg", "trust-exact", "trust-krylov", "trust-ncg"]

# These methods do not support bounds.
no_bounds = ["BFGS", "CG", "dogleg", "Newton-CG", "trust-exact",
             "trust-krylov", "trust-ncg"]

# These methods do not support constraints.
no_constraints = ["BFGS", "CG", "dogleg", "L-BFGS-B", "Nelder-Mead",
                  "Newton-CG", "Powell", "TNC", "trust-exact", "trust-krylov",
                  "trust-ncg"]