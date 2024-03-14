"""
Use pulp to solve the LP:

    minimize x + 2*y - z/2
    s.t.     x >= 0
             y >= 0
             z >= 0
             x +   y + z  = 1
                   z     <= 0.75
             x + 3*y     >= 1

"""

import pulp


x = pulp.LpVariable('x', lowBound=0)
y = pulp.LpVariable('y', lowBound=0)
z = pulp.LpVariable('z', lowBound=0)

prob = pulp.LpProblem('example2', pulp.LpMinimize)
prob += x + 2*y - 0.5*z   # Objective
prob += x + y + z == 1.0  # Equality constraint
prob += z <= 0.75         # Inequality constraint
prob += x + 3.0*y >= 1.0  # Inequality constraint

status = prob.solve()

print('Result:')
print(f'  status: {pulp.LpStatus[status]}')
print(f'  x = {pulp.value(x)}')
print(f'  y = {pulp.value(y)}')
print(f'  z = {pulp.value(z)}')
print(f'  objective = {pulp.value(prob.objective)}')
