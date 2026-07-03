import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from minimize_info import methods, require_jac, no_bounds, no_constraints
from foo import foo, foo_strict_bounds, foojac


b = Bounds(lb=[1.25, 2.0], ub=[1.3, 3.0])

lb0 = 1.25
width = 0.2
bf = Bounds(lb=[lb0, 2.0], ub=[lb0 + width, 3.0], keep_feasible=True)


print()
print("===== with Bounds(...) =====")
x0 = np.array([1.29, 2.75])
for method in methods:
    if method in no_bounds:
        print(f"{method:13} skipped: does not use bounds")
        continue
    jac = foojac if method in require_jac else None
    r = minimize(foo, x0=x0, method=method, jac=jac, bounds=b)
    print(f"{method:13} {r.success}  x = {r.x.tolist()}")

method_opts = {
    'trust-constr': dict(gtol=1e-14, xtol=1e-12),
    'powell':       dict(ftol=1e-13, xtol=1e-14),
}

print()
print("===== with Bounds(..., keep_feasible=True) =====")
x0 = np.array([np.nextafter(bf.ub[0], 0.0), 2.75])
for method in methods:
    if method in no_bounds:
        print(f"{method:13} skipped: does not use bounds")
        continue
    jac = foojac if method in require_jac else None
    opts = method_opts.get(method, {})
    try:
        r = minimize(foo_strict_bounds, x0=x0, tol=1e-8, method=method, jac=jac, bounds=bf, args=(bf,), options=opts)
    except RuntimeError:
        print(f"{method:13} *** violated constraint")
    else:
        print(f"{method:13} {r.success = } {r.status = }  x = {r.x.tolist()}")
