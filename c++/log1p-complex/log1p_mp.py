# Functions for generating expected values of log1p(z)
# for different input types of z.

import numpy as np
from mpmath import mp


def longdouble80_to_mpf(x):
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


def mpf_to_longdouble80(x):
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


def mpz_to_clongdouble80(z):
    # z is assumed to be an mp.mpc instance.
    re = mpf_to_longdouble80(z.real)
    im = mpf_to_longdouble80(z.imag)
    return np.clongdouble(re + im*1j)


def clongdouble80_to_mpz(z):
    # x is assumed to be np.clongdouble.
    return mp.mpc(longdouble80_to_mpf(z.real), longdouble80_to_mpf(z.imag))


def clongdouble80_log1p_mp(z):
    """
    Compute log1p(z) where z is np.clongdouble, using mpmath.

    The intent is for this function to be used to generate test cases
    for np.log1p(z).
    """
    z = clongdouble80_to_mpz(z)
    w = mp.log1p(z)
    return mpz_to_clongdouble80(w)


def doubledouble_log1p_mp(x, y):
    """
    Compute log1p(z), where z = x + y*i, with x and y in double-double format.

    x and y must be tuples of length 2 contain the upper and lower parts
    of their double-double represention (i.e. x = (xhi, xlo)).

    The return value is an mp.mpc instance.  To check this value againts
    a value computed in C/C++, print the real and imaginary parts in a
    context where mp.prec = 106.  E.g.

        x = (-1.2500111249898924, 7.43809241772238e-17)
        y = (0.6666666666666666, 3.700743415417188e-17)
        w = doubledouble_log1p_mp(x, y)
        with mp.workprec(106):
            print(w.real)
            print(w.imag)
    """
    xhi, xlo = x
    yhi, ylo = y
    x_mp = mp.mpf(xhi) + mp.mpf(xlo)
    y_mp = mp.mpf(yhi) + mp.mpf(ylo)
    z = mp.mpc(x_mp, y_mp)
    w = mp.log1p(z)
    return w


def complex64_log1p_mp(z):
    """
    Compute log1p(z), where z is an instance of np.complex64.

    Returns an instance of np.complex64.
    """
    x = mp.mpf(float(z.real))
    y = mp.mpf(float(z.imag))
    z_mp = mp.mpc(x, y)
    w_mp = mp.log1p(z_mp)
    return np.complex64(w_mp)


def complex128_log1p_mp(z):
    """
    Compute log1p(z), where z is an instance of np.complex128.

    Returns an instance of np.complex128.
    """
    z_mp = mp.mpc(z)
    w_mp = mp.log1p(z_mp)
    return np.complex128(w_mp)


def log1p_mp(z):
    if isinstance(z, np.complex64):
        return complex64_log1p_mp(z)
    if isinstance(z, (np.complex128, complex)):
        return complex128_log1p_mp(z)
    if isinstance(z, np.complex256):
        fi = np.finfo(z)
        if fi.machep == -63:
            # long double is 80 bit extended precision.
            return clongdouble80_log1p_mp(z)
    msg = [f'z has type {type(z)}']
    if isinstance(z, np.complex256):
        msg.append(' (which is not 80 bit extended precision)')
    msg.append(". This type is not handled by log1p_mp.\n")
    msg.append("For IBM double-double format, use doubledouble_log1p_mp(x, y).")
    raise RuntimeError("".join(msg))

