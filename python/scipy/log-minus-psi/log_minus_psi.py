"""
Calculate log(x) - psi(x) accurately.
(Work in progress!)
"""

import numpy as np
from scipy.special import psi
from mpmath import mp


def log_minus_psi_mp(x):
    return mp.log(x) - mp.psi(0, x)


def generate_pade_approx(x0, pa_order):
    with mp.workdps(80):
        tc = mp.taylor(log_minus_psi_mp, x0, sum(pa_order))
        p_mp, q_mp = mp.pade(tc, *pa_order)
        p = [float(t) for t in p_mp]
        q = [float(t) for t in q_mp]
        P = np.polynomial.Polynomial(p)
        Q = np.polynomial.Polynomial(q)
        return lambda x: P(x - x0) / Q(x - x0)


def naive(x):
    return np.log(x) - psi(x)


def approx4(x):
    return (1/2 + (1/12 + (-1/120 + 1/(256*x**2))/x**2)/x)/x


def approx5(x):
    return (1/2 + (1/12 + (-1/120 + (1/252 - 1/(240*x**2))/x**2)/x**2)/x)/x


def approx6(x):
    # Good for x > 20
    return (1/2 + (1/12 + (-1/120 + (1/252  + (-1/240 + 1/(132*x**2))/x**2)/x**2)/x**2)/x)/x



def log_minus_psi(x):
    # For scalars only.
    if x > 20:
        return approx6(x)
    if x < 2:
        return naive(x)
