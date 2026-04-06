"""
This module defines the function `lehmer_mean().
"""

import numpy as np

# Properties
# * L(x, 0) is the harmonic mean
# * L(x, 1) is the arithmetic mean
# * L(x, p) = 1 / L(1/x, -p + 1)

def lehmer_mean(x, *, p, weights=None, axis=-1, keepdims=False):
    """
    Compute the Lehmer mean of the 1-d sequence `x` of positive values.

    See https://en.wikipedia.org/wiki/Lehmer_mean

    `x` can be multidimensional.  The `axis` argument determines the
    axis of `x` over which the mean is calculated.

    `p` must be a scalar.

    The behavior of this implemetation when `x` contains `inf` or `nan` is
    undefined.  It will problaby return `nan`, and might also generate a
    warning.
    """
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64)
    if np.any(x <= 0):
        raise ValueError('x contains 0 or negative values')

    if p == 1:
        return np.average(x, weights=weights, axis=axis, keepdims=keepdims)
    if p > 0:
        scale = np.max(x, axis=axis, keepdims=True)
        u = x / scale
        m_numer = np.average(u**p, weights=weights, axis=axis, keepdims=True)
        m_denom = np.average(u**(p - 1), weights=weights, axis=axis, keepdims=True)
    else:
        scale = np.min(x, axis=axis, keepdims=True)
        u = scale / x
        m_numer = np.average(u**-p, weights=weights, axis=axis, keepdims=True)
        m_denom = np.average(u**(-p + 1), weights=weights, axis=axis, keepdims=True)
    lm = scale * (m_numer / m_denom)
    if keepdims:
        return lm
    lm = np.squeeze(lm, axis=axis)
    # One final step to maintain consistency with the case p == 1 where
    # np.average() is used: if the result is a scalar array, convert it
    # to a regular scalar type.
    return lm[()] if lm.shape == () else lm
