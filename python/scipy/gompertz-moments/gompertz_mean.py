
import numpy as np
from scipy.special import expi


# This polynomial is part of the expression for the asymptotic mean
# of the Gompertz distribution.
_gompertz_mean_asymp_poly = np.polynomial.Polynomial([1, -1, 2, -6, 24, -120, 720])


@np.vectorize
def gompertz_mean(c):
    """
    Mean of the Gompertz distribution for shape parameter c.

    This is with loc=0 and scale=1.
    """
    if c < 0:
        return np.nan
    if c == 0:
        return np.inf
    if c < 700:
        return -np.exp(c) * expi(-c)
    # Use the asymptotic formula.
    r = 1/c
    return r * _gompertz_mean_asymp_poly(r)
