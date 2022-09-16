import numpy as np


def interp1du(x, x0, step, f):
    """
    Interpolate linearly, under the assumption that the values in `f` are
    defined on a uniformly spaced grid [x0, x0 + step, x0 + 2*step, ...].

    The implementation is based on code provided by Phillip M. Feldman
    in https://github.com/scipy/scipy/issues/10180.

    Parameters
    ----------
    x : sequence
        Values at which to interpolate the function values.
    x0 : float
        Left end of the interval for x values of the given function values.
    step : float
        Step size of the uniforma grid on which the given data is sampled.
        (This is the same as the sample period, or 1/fs where fs is the
        sample rate.)
    f : sequence
        The given function values to be interpolated.
    """
    quo, rem = np.divmod(x - x0, step)
    fraction = rem / step

    # Although the quotient values returned by `numpy.divmod` are integers,
    # the array dtype is floating point.  We must force the dtype to integer
    # so that the quotient values can be used for indexing:
    quo = quo.astype(int)

    # In the following, `left` and `right` are clipped to ensure that we
    # don't attempt to index f outside of [0, len(f)-1].  In effect, this
    # implements "nearest endpoint" extrapolation.
    maximum = len(f) - 1
    left = np.clip(quo, 0, maximum)
    right = np.clip(quo + 1, 0, maximum)
    result = fraction * f[right] + (1 - fraction) * f[left]

    return result
