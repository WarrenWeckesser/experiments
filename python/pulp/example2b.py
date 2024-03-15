"""
Use pulp to solve the LP:

    minimize x + 2*y - z/2
    s.t.     x >= 0
             y >= 0
             z >= 0
                       z <= 0.75
             x + 3*y     >= 1
             x +   y + z  = 1

"""

import numpy as np
import pulp

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# These are the numpy arrays from the corresponding CVXOPT version
# of this example.
#
# The variable names are chosen to match the names used
# in the docstring of cvxopt.solvers.lp.

# Objective function coefficients
c = np.array([1.0, 2.0, -0.5])

# Inequality constraint coefficients (G*x <= h)
G = np.array([[-1.0,  0.0,  0.0],
              [ 0.0, -1.0,  0.0],
              [ 0.0,  0.0, -1.0],
              [ 0.0,  0.0,  1.0],
              [-1.0, -3.0,  0.0]])
h = np.array([0.0, 0.0, 0.0, 0.75, -1.0])

# Equality constraint coefficients (A*x = b)
A = np.array([[1.0, 1.0, 1.0]])
b = np.array([1.0])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Note: In this example, the variables are not defined with bounds.
# Instead, the bounds are included in the array of inequality
# constraints.
vars = [pulp.LpVariable(f'x_{k+1}') for k in range(G.shape[1])]

prob = pulp.LpProblem('example2', pulp.LpMinimize)

# Define the objective function.
prob += pulp.lpDot(c, vars)

# Define the inequality constraints.
for row, bound in zip(G, h):
    prob += pulp.lpDot(row, vars) <= bound

# Define the equality constraints.
for row, value in zip(A, b):
    prob += pulp.lpDot(row, vars) == value


status = prob.solve()

print('Result:')
print(f'  status: {pulp.LpStatus[status]}')
for var in vars:
    print(f'  {var} = {var.value()}')
print(f'  objective = {pulp.value(prob.objective)}')
