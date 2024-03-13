"""
Use pulp to solve the LP:

    maximize x + y
    s.t.     x >= 0
             y >= 0
               x - y >= -3
             -2x - y >= -9

The expected answer is x = 2, y = 5.
"""

import pulp


x = pulp.LpVariable('x', lowBound=0)
y = pulp.LpVariable('y', lowBound=0)

prob = pulp.LpProblem('example1', pulp.LpMaximize)
prob += x + y            # Objective
prob += x - y >= -3      # Constraint
prob += -2*x - y >= -9   # Constraint

status = prob.solve()

print('Result:')
print(f'  status: {pulp.LpStatus[status]}')
print(f'  x = {pulp.value(x)}')
print(f'  y = {pulp.value(y)}')
