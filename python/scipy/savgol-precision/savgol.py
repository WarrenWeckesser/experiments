# This code is from comments in
#    https://github.com/scipy/scipy/issues/20825
# by Niklas Z (MothNik on github).

### Imports ###

import numpy as np
from matplotlib import pyplot as plt
from scipy._lib._util import float_factorial
from scipy.linalg import lstsq

# NOTE: the following import name is to avoid confusion; it has no judgemental meaning
from scipy.signal import savgol_coeffs as unstable_savgol_coeffs

### Definition of a stablilised function ###

def stabilised_savgol_coeffs(
    window_length,
    polyorder,
    deriv=0,
    delta=1.0,
    pos=None,
    use="conv",
):
    # An alternative method for finding the coefficients when deriv=0 is
    #    t = np.arange(window_length)
    #    unit = (t == pos).astype(int)
    #    coeffs = np.polyval(np.polyfit(t, unit, polyorder), t)
    # The method implemented here is faster.

    # To recreate the table of sample coefficients shown in the chapter on
    # the Savitzy-Golay filter in the Numerical Recipes book, use
    #    window_length = nL + nR + 1
    #    pos = nL + 1
    #    c = savgol_coeffs(window_length, M, pos=pos, use='dot')

    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length.")

    halflen, rem = divmod(window_length, 2)

    if pos is None:
        if rem == 0:
            pos = halflen - 0.5
        else:
            pos = halflen

    if not (0 <= pos < window_length):
        raise ValueError("pos must be nonnegative and less than " "window_length.")

    if use not in ["conv", "dot"]:
        raise ValueError("`use` must be 'conv' or 'dot'")

    if deriv > polyorder:
        coeffs = np.zeros(window_length)
        return coeffs

    # Form the design matrix A. The columns of A are powers of the integers
    # from -pos to window_length - pos - 1. The powers (i.e., rows) range
    # from 0 to polyorder. (That is, A is a vandermonde matrix, but not
    # necessarily square.)
    x = np.arange(-pos, window_length - pos, dtype=float)
    # NOTE: x_min = - pos
    # NOTE: x_max = window_length - pos - 1
    x_abs_max = max(pos, window_length - pos - 1)
    # NOTE: the following avoid zero division in case `polyorder=0` and
    #       `window_length=1`
    scale = max(x_abs_max, 1.0)
    x /= scale
    delta *= scale

    if use == "conv":
        # Reverse so that result can be used in a convolution.
        x = x[::-1]

    order = np.arange(polyorder + 1).reshape(-1, 1)
    A = x**order

    # y determines which order derivative is returned.
    y = np.zeros(polyorder + 1)
    # The coefficient assigned to y[deriv] scales the result to take into
    # account the order of the derivative and the sample spacing.
    y[deriv] = float_factorial(deriv) / (delta**deriv)

    # Find the least-squares solution of A*c = y
    coeffs, _, _, _ = lstsq(A, y)

    return coeffs


# def simple_savgol_coeffs(window_length, polyorder):
#     if window_length % 2 != 1:
#         raise ValueError('window_length must be odd')
#     pos = window_length // 2
#     t = np.arange(window_length)
#     unit = (t == pos).astype(int)
#     coeffs = np.polyval(np.polyfit(t, unit, polyorder), t)
#     return coeffs


def simple_savgol_coeffs(window_length, polyorder, pos=None):
    if window_length % 2 != 1:
        raise ValueError('window_length must be odd')
    if pos is None:
        pos = window_length // 2
    t = np.arange(window_length)
    unit = (t == pos).astype(int)
    p = np.polynomial.Polynomial.fit(t, unit, deg=polyorder)
    coeffs = p(t)
    return coeffs[::-1]


def check_savgol_coeffs(c, cref):
    relerr = np.array([float(abs((c0 - cref0)/cref0)) if cref != 0 else np.inf
                       for c0, cref0 in zip(c, cref)])
    return relerr
