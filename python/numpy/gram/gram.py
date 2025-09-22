import numpy as np


def gram(x, *, M=None, w=None):
    """
    Compute the Gram matrix.

    The Gram matrix is the matrix of inner products of the columns of `x`.
    By default, the standard Euclidean inner product <x, x> is computed.

    When `M` is provided, the inner product is <x, M@x>.  So generally `M`
    should be Hermitian (symmetric in the real case), but this is not checked
    and does not affect the calculation.

    `x` must be at least 2-d, with shape `(..., m, n)`.

    If given, 'M' must be an array with shape `(..., m, m)`.
    This is, it can be a "batch" of 2-d arrays.
    
    If given, `w` must be an array with shape `(..., m)`.
    That is, it can be a "batch" of 1-d arrays.
    Using `w` is equivalent to using `M=np.diag(w)`.

    The result has shape `(..., n, n)`.
    """
    if w is not None and M is not None:
        raise ValueError('only one of `w` or `M` may be given')
    if np.iscomplexobj(x):
        y = x.conjugate().transpose()
    else:
        y = x.transpose()
    if w is None and M is None:
        return y @ x
    if M is None:
        return np.einsum('...ik,...k,...kj->...ij', y, w, x)
    # M is not None
    return np.einsum('...ik,...kl,...lj->...ij', y, M, x)
