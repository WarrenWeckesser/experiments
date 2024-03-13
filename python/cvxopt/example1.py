"""
Use cvxopt to solve the LP:

    maximize x + y
    s.t.     x >= 0
             y >= 0
               x - y >= -3
             -2x - y >= -9

The expected answer is x = 2, y = 5.
"""

from cvxopt import matrix, solvers

A = matrix([[-1.0, 0.0, -1.0, 2.0], [0.0, -1.0, 1.0, 1.0]])
b = matrix([0.0, 0.0, 3.0, 9.0])
c = matrix([-1.0, -1.0])

sol = solvers.lp(c, A, b)
print(sol['x'])
