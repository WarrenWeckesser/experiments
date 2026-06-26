import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from minimize_info import methods, require_jac, no_bounds, no_constraints
from foo import foo, foojac


# This constraint is x0 >= 2.
b = Bounds(lb=[2.0, -np.inf], ub=[np.inf, np.inf])

# This constraint is x0 + x1 >= 3.
linear_constr = LinearConstraint(A=[[1.0, 1.0]], lb=3, ub=np.inf)

# This constraint is x1 > exp(x0)
nonlinear_constr = NonlinearConstraint(lambda x: np.exp(x[0]) - x[1],
                                       lb=-np.inf, ub=0)

print()
print("===== with Bounds (x0 > 2) =====")
x0 = np.array([2.25, -0.25])
for method in methods:
    if method in no_bounds:
        print(f"{method:13} skipped: does not use bounds")
        continue
    jac = foojac if method in require_jac else None
    r = minimize(foo, x0=x0, method=method, jac=jac, bounds=b)
    print(f"{method:13} {r.success}  x = {r.x.tolist()}")

print()
print("===== with LinearConstraint (x0 + x1 >= 3) =====")
x0 = np.array([3.25, 0.125])
for method in methods:
    if method in no_constraints:
        print(f"{method:13} skipped: does not use constraints")
        continue
    jac = foojac if method in require_jac else None
    r = minimize(foo, x0=x0, method=method, jac=jac, constraints=linear_constr)
    print(f"{method:13} {r.success}  x = {r.x.tolist()}")

print()
print("===== with NonlinearConstraint (x1 >= exp(x0)) =====")
x0 = np.array([1.0, 3.0])
for method in methods:
    if method in no_constraints:
        print(f"{method:13} skipped: does not use constraints")
        continue
    jac = foojac if method in require_jac else None
    r = minimize(foo, x0=x0, method=method, jac=jac, constraints=nonlinear_constr)
    print(f"{method:13} {r.success}  x = {r.x.tolist()}")