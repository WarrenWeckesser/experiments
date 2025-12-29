# See https://stackoverflow.com/questions/79851453/count-common-values-between-different-2d-arrays

import numpy as np


def bincount_1d_slices(a, maxvalue):
    """
    `a` is assumed to be an array of integers.
    """
    return np.apply_along_axis(np.bincount, -1, a, minlength=maxvalue + 1)


def intersection_size(x, y, maxvalue=None):
    """
    Compute the sizes of the intersections of the rows of `x` and `y`.

    x and y are assumed to be 2-d integer arrays.

    If maxvalue is not given, the maximum of the values in x and y is used.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if maxvalue is None:
        maxvalue = int(max(np.max(x), np.max(y)))
    xcount = bincount_1d_slices(x, maxvalue)
    ycount = bincount_1d_slices(y, maxvalue)
    return np.sum(np.logical_and(xcount, np.expand_dims(ycount, 1)), axis=-1)
