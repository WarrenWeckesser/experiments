"""
An example of the calculation behind isotonic regression
(https://en.wikipedia.org/wiki/Isotonic_regression).

This script uses `scipy.optimize.minimize` with a constraint function
to compute the sequence of monotonically increasing "y-hat" values.
This is not the most efficient method for computing these values!
This script is a demonstration of the idea; it is not a basis for an
efficient implementation.

This script shows the results of scipy.optimize.isotonic_regression
and the equivalent constrained minimization.
"""

import numpy as np
from scipy.optimize import minimize, isotonic_regression
from scipy.interpolate import interp1d, PchipInterpolator
import matplotlib.pyplot as plt

y = np.array([1.5, 1.0, 4.0, 6.0, 5.7, 5.0, 7.8, 9.0, 7.5, 9.5, 9.0])
x = np.arange(len(y))


def objective(yhat, y):
    return np.sum((yhat - y)**2)


def constraint(yhat, y):
    # This is for a monotonically increasing regression.
    return np.diff(yhat)


result = minimize(objective, x0=y, args=(y,),
                  constraints=[{'type': 'ineq',
                                'fun': lambda x: constraint(x, y)}])

yhat = result.x
if result.success:
    print("Constrained minimization:")
    print(yhat)
else:
    print("*** minimize failed ***")
    print(result)
    exit(-1)

ir = isotonic_regression(y)
print("SciPy: ")
print(ir.x)
