"""
An example of the calculation behind isotonic regression
(https://en.wikipedia.org/wiki/Isotonic_regression).

This script uses `scipy.optimize.minimize` with a constraint function
to compute the sequence of monotonically increasing "y-hat" values.
This is not the most efficient method for computing these values!
This script is a demonstration of the idea; it is not a basis for an
efficient implementation.

The code doesn't actually create an interpolator, but the green line
in the plot shows the graph of what the interpolator would be if linear
interpolation is used between the compute isotonic points.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


data = np.array([[1.2, 1.5],
                 [2.0, 1.0],
                 [3.0, 4.0],
                 [5.0, 6.0],
                 [5.5, 5.7],
                 [6.5, 5.0],
                 [7.5, 7.8],
                 [8.0, 9.0],
                 [8.5, 7.5],
                 [9.0, 9.5],
                 [9.5, 9.0]])

x, y = data.T


def objective(yhat, y):
    return np.sum((yhat - y)**2)


def constraint(yhat, y):
    # This is for a monotonically increasing regression.
    return np.diff(yhat)


result = minimize(objective, x0=y, args=(y,),
                  constraints=[{'type': 'ineq',
                                'fun': lambda x: constraint(x, y)}])

print(result)
yhat = result.x

plt.plot(x, y, 'k.', alpha=0.75, label='data')
plt.plot(x, yhat, 'go-', alpha=0.4, label='isotonic + linear interp')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.legend(framealpha=1, shadow=True)

plt.show()
