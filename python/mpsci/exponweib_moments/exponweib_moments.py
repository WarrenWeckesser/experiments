"""
Raw moment calculation for the exponentiated Weibull distribution.

The function raw_moment(k, theta, c, alpha) uses mpmath to implement
the raw_moment calculation for the expoentiated Weibull distribution,
as given in [1].


..[1] Choudhury, A. (2005). "A Simple Derivation of Moments of the Exponentiated
      Weibull Distribution". Metrika. 62 (1): 17-22. doi:10.1007/s001840400351
"""

from mpmath import mp


def a_coeff(i, theta):
    """
    See equation (3.2) of [1].
    """
    y = 1
    for j in range(i):
        y *= (theta - 1 - j)
    return mp.power(-1, i) * y / mp.factorial(i)


def term(i, c, k, theta):
    """
    Compute the i-th term of the infinite series in equation (3.3) of [1].
    """
    i = int(i)
    c = mp.mpf(c)
    return a_coeff(i, theta) * mp.power(i + 1, -(k/c + 1))


def raw_moment(k, theta, c, alpha):
    """
    Compute the raw moment of the exponentiated Weibull distribution.

    The function uses mpmath.nsum to implement equation (3.3) of the paper [1].
    The parameter names match those of the paper.

    Parameter mapping between here and mpsci.distributions.exponweib:

        here    mpsci
        -----   -----
        theta   a
        c       c
        alpha   scale
    """
    theta = mp.mpf(theta)
    c = mp.mpf(c)
    alpha = mp.mpf(alpha)
    s = mp.nsum(lambda i: term(i, c, k, theta), [1, mp.inf], steps=[100, 10], method='l')
    return theta * mp.power(alpha, k) * mp.gamma(k/c + 1) * (1 + s)
