#
# Experiment with maximum likelihood fit for the negative
# binomial distribution.
#

import numpy as np
from scipy.special import digamma, gammaln, xlogy, xlog1py
from scipy.optimize import fsolve


def loglik(r, p, samples):
    samples = np.asarray(samples)
    N = len(samples)
    t1 = gammaln(samples + r).sum()
    t2 = gammaln(samples + 1).sum()
    t3 = N*gammaln(r)
    t4 = xlogy(samples, p).sum()
    t5 = N*xlog1py(r, -p)
    return t1 - t2 - t3 + t4 + t5


def func(r, samples):
    samples = np.asarray(samples)
    N = len(samples)
    s = samples.sum()
    dsum = digamma(samples + r).sum()
    q = np.log(r / (r + s/N))
    return dsum - N*digamma(r) + N*q


def nbinom_mle(samples):
    """
    Maximum likelihood estimation for the negative binomial distribution.

    Returns *four* values: r, p, r_int, p_int

    The pair (r, p) is the MLE under the assumption that r is not
    restricted to integers values.  The pair (r_int, p_int) is the
    MLE when r is constrained to be an integer.
    """
    r = fsolve(func, 1, samples, xtol=1e-10)[0]
    s = samples.sum()
    p = s / (len(samples)*r + s)
    rint = int(r)
    if rint != r:
        if loglik(rint, p, samples) < loglik(rint + 1, p, samples):
            rint += 1
    m = np.mean(samples)
    return r, p, rint, m/(rint + m)


samples = np.array([3, 5, 2, 10, 21, 13, 5, 4, 7, 7])
# samples = np.array([1, 2, 2, 3, 5, 2, 10, 11, 13, 5, 4, 7, 7])

r, p, rint, p1 = nbinom_mle(samples)
print(f"{r = }  {p = }")
print(f"{rint = }  {p1 = }")


# Compare to R:
# > library(fitdistrplus)
# > samples = c(3, 5, 2, 10, 21, 13, 5, 4, 7, 7)
# > fit = fitdist(samples, "nbinom")
# > fit
# Fitting of the distribution ' nbinom ' by maximum likelihood
# Parameters:
#      estimate Std. Error
# size 3.361170   2.086297
# mu   7.698005   1.591145
# > p = fit$estimate['mu']/(fit$estimate['size'] + fit$estimate['mu'])
# > p
#        mu
# 0.6960741
#
# My calculation matches R to only three significant digits.
# What error tolerances does R use to find the parameters?
#
# With the mledist method (from fitdistrplus), using optim.method set
# to "BFGS", "CG" or "L-BFGS-B", the result matches the Python calculation:
#
# > mledist(samples, "nbinom", optim.method="BFGS")
# $estimate
#     size       mu
# 3.360125 7.700000
# <snip>
#
# > mledist(samples, "nbinom", optim.method="CG")
# $estimate
#    size      mu
# 3.36012 7.70000
# <snip>
#
# > mledist(samples, "nbinom", optim.method="L-BFGS-B")
# $estimate
#     size       mu
# 3.360127 7.700000
# <snip>
