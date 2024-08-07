"""
Some functions for the "standard" Cauchy distribution (loc=0, scale=1).
"""
import numpy as np


def cauchy_cdf(x):
    return -np.arctan2(-1, -x)/np.pi


def cauchy_sf(x):
    return -np.arctan2(-1, x)/np.pi


# This works great for p < 0.5.  It loses precision as p -> 1.
def cauchy_invcdf(p):
    return -1/np.tan(np.pi*p)


# This works great for p > 0.5.  It loses precision as p -> 0.
def cauchy_invsf(p):
    return 1/np.tan(np.pi*p)
