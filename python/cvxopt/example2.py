"""
Use cvxopt to solve the LP:

    minimize x + 2*y - z/2
    s.t.     x >= 0
             y >= 0
             z >= 0
             x +   y + z  = 1
                   z     <= 0.75
             x + 3*y     >= 1

"""

import numpy as np
from cvxopt import matrix, solvers


# The variables names are chosen to match the names used
# in the docstring of cvxopt.solvers.lp.

# Object function coefficients
c = matrix([1.0, 2.0, -0.5])

# G = matrix([[-1.0,  0.0,  0.0, -1.0, 0.0],
#             [ 0.0, -1.0,  0.0, -3.0, 0.0],
#             [ 0.0,  0.0, -1.0,  0.0, 1.0]])
# h = matrix([0.0, 0.0, 0.0, 1.0, 0.75])

G = matrix([[-1.0,  0.0,  0.0, 0.0, -1.0],
            [ 0.0, -1.0,  0.0, 0.0, -3.0],
            [ 0.0,  0.0, -1.0, 1.0,  0.0]])
h = matrix([0.0, 0.0, 0.0, 0.75, -1.0])

A = matrix([[1.0], [1.0], [1.0]])
b = matrix([1.0])

solvers.options['maxiters'] = 500
solvers.options['abstol'] = 1e-14
solvers.options['reltol'] = 1e-10
solvers.options['feastol'] = 4e-10

sol = solvers.lp(c=c, G=G, h=h, A=A, b=b)
w = np.array(sol['x'])
print(w)
print(f'objective = {sol['primal objective']}')
