"""
Use highspy to solve the LP:

    maximize x + y
    s.t.     x >= 0
             y >= 0
               x - y >= -3
             -2x - y >= -9

The expected answer is x = 2, y = 5.
"""

import numpy as np
import highspy

# Highs h
h = highspy.Highs()

# Two nonnegative variables.
inf = highspy.kHighsInf
h.addVars(2, np.array([0.0, 0.0]), np.array([inf, inf]))

# Objective function coefficients.
h.changeColsCost(2, np.array([0, 1]), np.array([-1.0, -1.0], dtype=np.double))

# Configure the constraints.
num_cons = 2
lower = np.array([-3.0, -9.0], dtype=np.double)
upper = np.array([inf, inf], dtype=np.double)
num_new_nz = 4
starts = np.array([0, 2])
indices = np.array([0, 1, 0, 1])
values = np.array([1, -1, -2, -1], dtype=np.double)
h.addRows(num_cons, lower, upper, num_new_nz, starts, indices, values)

h.run()
sol = h.getSolution()

print()
print(f'status:   {h.modelStatusToString(h.getModelStatus())}')
print(f'solution: {sol.col_value}  (expected solution: [2, 5])')
