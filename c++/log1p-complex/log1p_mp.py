# Functions for generating reference values of log1p(z)
# for different input types of z.
#
# This code has not been tested on platforms where long double is
# IEEE float128 or IBM double-double.  In the case of IBM double-double,
# the function doubledouble_log1p_from_xy_mp(x, y) can be used to
# generate test cases by passing in the already split x and y components
# of the np.clongdouble instance.

import numpy as np
from mpmath import mp


def mpf_to_longdouble(x):
    """
    Convert an mp.mpf instance to a np.longdouble.
    """
    machep = np.finfo(np.longdouble).machep
    if machep == -52:
        # long double is the same as double.
        return np.longdouble(float(x))
    prec = 1 - machep
    with mp.workprec(prec):
        man, exp = mp.mpf(x).man_exp
    sum = np.longdouble(0)
    for k in range(prec):
        n = prec - 1 - k
        if man & (1 << n):
            sum += np.longdouble(2)**(n + exp)
    return sum


def mpz_to_clongdouble(z):
    x = mpf_to_longdouble(z.real)
    y = mpf_to_longdouble(z.imag)
    return np.clongdouble(x + y*1j)


def longdouble_to_mpf(x):
    """
    Convert a np.longdouble instance to mp.mpf.

    x must be an instance of np.longdouble.
    """
    fi = np.finfo(x)
    if fi.machep == -52:
        # long double is the same as double.
        return mp.mpf(float(x))
    if fi.machep == -63:
        # Special case: np.longdouble is 80 bit extended precision.
        man, p = np.frexp(x)
        u = x.view(np.dtype((np.uint64, 2)))[0]
        return np.sign(x)*mp.mpf((u.item(0), int(p) - 64))
    prec = 1 - fi.machep
    with mp.workprec(prec):
        # XXX I don't know how safe this is. It might not get the ULP correct.
        x_mp = mp.mpf(str(x))
    return x_mp


def clongdouble_to_mpz(z):
    """
    Convert an instance of np.clongdouble to mp.mpf.
    """
    return mp.mpc(longdouble_to_mpf(z.real), longdouble_to_mpf(z.imag))


def doubledouble_log1p_from_xy_mp(x, y):
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


def split_doubledouble(x):
    """
    x must be an instance of np.longdouble.

    Use this function only when the platform implements np.longdouble
    using the IBM double-double format.
    """
    if np.finfo(np.longdouble).machep != -105:
        raise RuntimeError('np.longdouble is not implemented as double-double')
    xhi, xlo = np.array([x]).view(np.float64)
    return (xhi, xlo)


def doubledouble_log1p_mp(z):
    """
    Compute log1p(z) when np.longdouble uses double-double format.
    """
    xhilo = split_doubledouble(z.real)
    yhilo = split_doubledouble(z.imag)
    w_mp = doubledouble_log1p_from_xy_mp(xhilo, yhilo)
    return mpz_to_clongdouble(w_mp)


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


def clongdouble_log1p_mp(z):
    """
    Compute log1p(z), where z is an instance of np.clongdouble.

    Returns an instance of np.clongdouble.

    Warning: This function has not been tested for the case where
    np.longdouble is IEEE float128.
    """
    z_mp = clongdouble_to_mpz(z)
    w_mp = mp.log1p(z_mp)
    return mpz_to_clongdouble(w_mp)


def log1p_mp(z):
    """
    Compute log1p(z) for complex z.

    z must an instance of a numpy complex type or a Python complex().

    Intermediate calculations use mpmath.  Be sure mpmath.mp.dps is set
    sufficiently high so there is no doubt that the mpmath calculations
    maintain enough precision for the final output.
    """
    if isinstance(z, np.complex64):
        return complex64_log1p_mp(z)
    if isinstance(z, (np.complex128, complex)):
        return complex128_log1p_mp(z)
    if isinstance(z, np.clongdouble):
        fi = np.finfo(z)
        if fi.machep == -52:
            # long double is the same as double.
            return np.clongdouble(complex128_log1p_mp(np.cdouble(z)))
        if fi.machep == -63 or fi.machep == -112:
            # 80 bit extended precision or IEEE float128
            return clongdouble_log1p_mp(z)
        if fi.machep == -105:
            # long double is IBM double-double.
            return doubledouble_log1p_mp(z)
        msg = ("z is a np.clongdouble, but the underlying floating point "
               "format is not handled by this function.")
        raise RuntimeError(msg)
    raise TypeError('z is not a numpy complex type.')
