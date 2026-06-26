import numpy as np
from scipy.optimize import minimize
from minimize_info import methods, no_jac, require_jac, no_hess, require_hess
from foo import foo, foojac, foohess


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

x0 = np.array([2.25, -0.25])

print("===== basic =====")
for method in methods:
    if method in require_jac:
        print(f"{method:13} skipped: requires jac")
        continue
    r = minimize(foo, x0=x0, method=method)
    print(f"{method:13} {r.success}  x = {r.x.tolist()}")

print()
print("===== with jac =====")
for method in methods:
    if method in require_hess:
        print(f"{method:13} skipped: requires hess")
        continue
    if method in no_jac:
        print(f"{method:13} skipped: does not use jac")
        continue
    r = minimize(foo, x0=x0, method=method, jac=foojac)
    print(f"{method:13} {r.success}  x = {r.x.tolist()}")

print()
print("===== with jac and hess =====")
for method in methods:
    if method in no_jac:
        print(f"{method:13} skipped: does not use jac")
        continue
    if method in no_hess:
        print(f"{method:13} skipped: does not use hess")
        continue
    r = minimize(foo, x0=x0, method=method, jac=foojac, hess=foohess)
    print(f"{method:13} {r.success}  x = {r.x.tolist()}")