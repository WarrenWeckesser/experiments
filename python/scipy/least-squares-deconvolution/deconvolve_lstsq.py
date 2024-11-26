import numpy as np
from scipy.linalg import toeplitz


def deconvolve_lstsq(x, y, symmetry=None):
    """
    Given y = convolve(x, h), find the least-squares solution for h.

    x and y are one-dimensional arrays.

    The length of h is inferred to be len(y) - len(x) + 1.

    If symmetry is "even", constrain h to have even symmetry.

    This is a simple (naive) implementation.  It has been tested with only
    a few examples, and with len(h) small (and len(h) much less than len(x)).

    See Section 8.4 of Ivan Selesnick's notes on "Least Squares with
    Examples in Signal Processing" for information on least-squares
    deconvolution.
    """
    if symmetry not in [None, "even"]:
        raise ValueError("unknown symmetry option %r" % (symmetry,))

    m = len(y) - len(x) + 1
    zpad = np.zeros(m-1)
    xp = np.r_[x, zpad]

    T = toeplitz(xp, np.zeros(m))

    if symmetry is None:
        # Coefficients of h are unconstrained.
        h, resid, rnk, sv = np.linalg.lstsq(T, y, rcond=None)
    else:
        # Enforce even symmetry in h.
        hm, r = divmod(m, 2)
        e = np.eye(hm + r)
        K = np.concatenate((e, e[hm - r::-1]), axis=0)
        TK = T.dot(K)
        h_symm, resid, rnk, sv = np.linalg.lstsq(TK, y, rcond=None)
        h = np.concatenate((h_symm, h_symm[hm - r::-1]))

    return h
