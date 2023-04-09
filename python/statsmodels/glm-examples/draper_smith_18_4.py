"""
Use statsmodels to compute the result shown in section 18.4 of
*Applied Regression Analysis* (third edition, 1998) by Draper
and Smith.

It is an application of the generalized linear model, using
the Poisson family and the natural (i.e. log) link.
"""

import numpy as np
import statsmodels.api as sm


# Load the data from the file.
a, b, t = np.loadtxt('draper_smith_18_4_data.txt', skiprows=1, unpack=True)

# Reshape into the form needed by sm.GLM().
y = np.concatenate((a, b))
x = np.tile(np.log(t), 2)
w = np.concatenate((np.zeros_like(a), np.ones_like(b)))

# Prepend column of 1s for the constant term, and form X,
# the array of independent (a.k.a. exogenous) variables.
X = np.column_stack((np.ones_like(x), x, w))

glm = sm.GLM(y, X, family=sm.families.Poisson())
res = glm.fit()
print(res.summary())

print()
print("Computed coefficients: %10.5f %10.5f %10.5f" % tuple(res.params))
print("From the text:            1.684      0.3784     0.6328")
