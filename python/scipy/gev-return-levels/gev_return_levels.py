"""
This script provides an example of computing return levels for several
return periods based on a fit of the generalized extreme value (GEV)
distribution to a dataset of annual maximum temperatures.

See https://stackoverflow.com/questions/71202562/
    calculating-return-value-using-generalised-extreme-value-distribution-gev
"""

import numpy as np
from scipy.optimize import fmin
from scipy.stats import genextreme


def mymin(func, x0, args=None, disp=False):
    # This function is used in genextreme.fit() below.  It is not essential;
    # the code below works well enough with the default optimizer.  This
    # function allows the optimizer to be fine-tuned, which can be helpful
    # when comparing the result of genextreme.fit() to the results of other
    # software packages.
    return fmin(func, x0, args=args, disp=disp,
                xtol=1e-9, ftol=1e-10,
                maxiter=10000)


# Annual maximum temperatures
data = np.array([28.01, 29.07, 28.67, 21.57, 21.66, 24.62, 21.45, 28.51,
                 22.65, 21.57, 20.89, 20.96, 21.05, 22.29, 20.81, 21.08,
                 20.77, 23.18, 22.98, 21.88, 21.07, 20.74, 22.69, 22.42,
                 31.81, 25.78, 29.09, 28.11, 22.18, 21.6])

# Fit the generalized extreme value distribution to the data.
shape, loc, scale = genextreme.fit(data, optimizer=mymin)
print("GEV fit parameters:")
print(f"  shape: {shape:.5f}")
print(f"  loc:   {loc:.5f}")
print(f"  scale: {scale:.5f}")
print()

# Compute the return levels for several return periods.
return_periods = np.array([5, 10, 20, 50, 100])
return_levels = genextreme.isf(1/return_periods, shape, loc, scale)

print("Return levels:")
print()
print("Period    Level")
print("(years)   (temp)")

for period, level in zip(return_periods, return_levels):
    print(f'{period:4.0f}  {level:9.2f}')
