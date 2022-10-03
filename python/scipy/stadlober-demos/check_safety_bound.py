import math
import numpy as np
from scipy.stats import hypergeom


def hypergeom_meanvar(ngood, nbad, nsample):
    total = ngood + nbad
    mu = nsample * ngood / total
    var = mu * (nbad/total) * (total - nsample)/(total - 1)
    return mu, var


ngood = 1000
nbad = 1500
total = ngood + nbad
nsample = 250
mu, var = hypergeom_meanvar(ngood, nbad, nsample)

# slen is the support length.  It is the largest value of k for which
# the probability of hypergeom.pmf(slen-1, total, ngood, nsample) is
# nonzero.
slen = min(nsample, ngood) + 1

# B is (roughly) the number of standard deviations beyond the mean
# of the distribution at which the probability values are considered
# negligible.
B = 16
ub = math.floor(mu + B*math.sqrt(var + 0.5))

p = hypergeom.pmf(np.arange(slen), total, ngood, nsample)

print("ngood =", ngood, "  nbad =", nbad, "   total =", total)
print("mu =", mu, "   var =", var)
print("len(support) =", slen)
print("upper bound  =", ub)
print("p[-1] =", p[-1])
if ub < slen:
    print("p(ub) =", p[ub])
