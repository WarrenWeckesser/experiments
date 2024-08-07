"""
Some functions for the "standard" Cauchy distribution (loc=0, scale=1).
"""
import numpy as np


def cauchy_cdf(x):
    return -np.arctan2(-1, -x)/np.pi


def cauchy_sf(x):
    return -np.arctan2(-1, x)/np.pi


def cauchy_invcdf(p, loc=0, scale=1):
    return -1/np.tan(np.pi*p)


def cauchy_invsf(p, loc=0, scale=1):
    return 1/np.tan(np.pi*p)
