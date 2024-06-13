
import numpy as np
from mpmath import mp


def longdouble_to_mpf(x):
    """
    Convert np.longdouble (80 bit extended precision) to mp.mpf.

    mp.mpf(x) does not work if x is np.longdouble.  This function
    will create an mp.mpf instance that matches the full precision
    of x (assuming mp.dps is sufficiently large).

    x must be an instance of np.longdouble with 80 bit extended
    precision format.
    """
    if np.finfo(x).machep != -63:
        raise ValueError('x must be an instance of np.longdouble where the '
                         'longdouble type is 80 bit extended precision.')
    man, p = np.frexp(x)
    u = x.view(np.dtype((np.uint64, 2)))[0]
    return np.sign(x)*mp.mpf((u.item(0), int(p) - 64))


def mpf_to_longdouble(x):
    """
    Convert an mp.mpf instance to a np.longdouble.

    This only works on platforms where np.longdouble is actually 80 bit
    extended precision floating point.

    Note that if the current precision of x is greater than 64, this
    conversion is lossy.
    """
    # This function is for 80 bit extended precision format.
    if np.finfo(np.longdouble).machep != -63:
        raise RuntimeError("np.longdouble is not 80 bit extended precision")
    with mp.workprec(64):
        man, exp = mp.mpf(x).man_exp
    sum = np.longdouble(0)
    for k in range(64):
        n = 63 - k
        if man & (1 << n):
            sum += np.longdouble(2)**(n + exp)
    return sum


def mpz_to_clongdouble(z):
    # z is assumed to be an mp.mpc instance.
    re = mpf_to_longdouble(z.real)
    im = mpf_to_longdouble(z.imag)
    return np.clongdouble(re + im*1j)


def clongdouble_to_mpz(z):
    # x is assumed to be np.clongdouble.
    return mp.mpc(longdouble_to_mpf(z.real), longdouble_to_mpf(z.imag))


def clongdouble_log1p_mp(z):
    """
    Compute log1p(z) where z is np.clongdouble, using mpmath.

    The intent is for this function to be used to generate test cases
    for np.log1p(z).
    """
    z = clongdouble_to_mpz(z)
    w = mp.log1p(z)
    return mpz_to_clongdouble(w)
