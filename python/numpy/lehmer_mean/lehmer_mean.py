import numpy as np


def lehmer_mean(x, *, p, weights=None, axis=-1, keepdims=False):
    """
    Compute the Lehmer mean of the 1-d sequence `x` of nonnegative values.

    See https://en.wikipedia.org/wiki/Lehmer_mean

    `x` can be multidimensional.  The `axis` argument determines the
    axis of `x` over which the mean is calculated.

    `p` is expected to be a scalar.  Multidimensional `p` might work if
    its shape is compatible for broadcasting with `x`, but this is not
    tested.
    """
    x = np.asarray(x)
    if np.any(x < 0):
        raise ValueError('x contains negative values')
    # x is scaled by xmax to create u.  This avoids overflow and underflow
    # in examples such as lehmer_mean([1e181, 1e175, 3e180], p=2) and
    # lehmer_mean([2e-100, 1e-101, 3.5e-100], p=4).
    xmax = np.max(x, axis=axis, keepdims=True)
    u = x / xmax
    m_numer = np.average(u**p, weights=weights, axis=axis, keepdims=True)
    m_denom = np.average(u**(p - 1), weights=weights, axis=axis, keepdims=True)
    lm = xmax * (m_numer / m_denom)
    if keepdims:
        return lm
    return np.squeeze(lm, axis=axis)
