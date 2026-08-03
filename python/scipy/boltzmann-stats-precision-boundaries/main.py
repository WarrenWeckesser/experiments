
import math
from mpmath import mp
from mpsci.distributions import trunc_discrete_exp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boltzmann


mp.dps = 100


def find_mean_precision_boundary(n, threshhold=5e-15):
    lam = 1e-16
    count = 0
    r = 1.125
    while count < 20:
        lam = r*lam
        mean_mp = trunc_discrete_exp.mean(lam, n)
        mean = boltzmann.mean(lam, n)
        relerr = float(abs((mean - mean_mp)/mean_mp))
        if relerr < threshhold:
            if count == 0:
                result = lam
            count += 1
        else:
            count = 0
    return result


def trunc_discrete_exp_mean_small_lam(lam, n):
    """
    Approximate mean of the truncated discrete exponential distribution.
    The approximation improves as lam -> 0.
    """
    return (n - 1) * ((n**3 + n**2 + n + 1)/720*lam**3 - (n + 1)*lam/12 + 0.5)
