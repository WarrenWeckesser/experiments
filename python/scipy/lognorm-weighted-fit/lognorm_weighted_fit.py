
import numpy as np
from scipy.stats import lognorm


# Sample data and weights.  To enable an exact comparison with
# the method of generating an array with value repeated according
# to their weight, I use an array of weights that is all integers.
x = np.array([2.5, 8.4, 9.3, 10.8, 6.8, 1.9, 2.0])
w = np.array([  1,   1,   2,    1,   3,   3,   1])


# Fit the log-normal distribution by creating an array containing the values
# repeated according to their weight.
xx = np.repeat(x, w)

# Use the explicit formulas for the MLE of the log-normal distribution.
lnxx = np.log(xx)
muhat = np.mean(lnxx)
varhat = np.var(lnxx)

shape = np.sqrt(varhat)
scale = np.exp(muhat)

print("MLE using repeated array: shape=%7.5f   scale=%7.5f" % (shape, scale))


# Use the explicit formulas for the weighted MLE of the log-normal
# distribution.

lnx = np.log(x)
muhat = np.average(lnx, weights=w)
# varhat is the weighted variance of ln(x).  There isn't a function in numpy
# for the weighted variance, so we compute it using np.average.
varhat = np.average((lnx - muhat)**2, weights=w)

shape = np.sqrt(varhat)
scale = np.exp(muhat)

print("MLE using weights:        shape=%7.5f   scale=%7.5f" % (shape, scale))


# Might as well check that we get the same result from lognorm.fit() using the
# repeated array

shape, loc, scale = lognorm.fit(xx, floc=0)

print("MLE using lognorm.fit:    shape=%7.5f   scale=%7.5f" % (shape, scale))
