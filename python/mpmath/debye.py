import numpy as np
from scipy import integrate
from mpmath import mp


def mp_debye1(x):
    if x == 0:
        return mp.one

    def integrand(t):
        if t == 0:
            return mp.one
        return t / mp.expm1(t)

    return mp.quad(integrand, [0, x])/x


_C = 1.6449340668482264  # pi**2/6


def debye1(x):
    if abs(x) < 0.08:
        # First few terms of the series expansion.
        return (1 + x*(-1/4 + x*(1/36 + x**2*(-1/3600 + x**2/211680))))
    if x > 40:
        # Asymptotic expansion.
        return _C/x
    if x < 0:
        return debye1(-x) - x/2

    def integrand(t):
        if t == 0:
            return 1.0
        return t / np.expm1(t)

    return integrate.quad(integrand, 0, x)[0] / x
