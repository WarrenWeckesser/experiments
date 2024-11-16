# https://stackoverflow.com/questions/53401040/
#     least-squares-using-convolution-in-python

import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import deconvolve


def deconvolve_lstsq(x, y, symmetry=None):
    """
    Given y = convolve(x, h), find the least-squares solution for h.

    x and y are one-dimensional arrays.

    The length of h is inferred to be len(y) - len(x) + 1.

    If symmetry is "even", constrain h to have even symmetry.

    This is a simple (naive) implementation.  It has been tested with only
    a few example, and with len(h) small (and len(h) much less than len(x)).

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


np.random.seed(1231231)

n = 400
#x = np.random.randint(-3, 10, n)
x = np.zeros(n)
num_impulses = 15
indices = np.random.randint(0, n, size=num_impulses)
x[indices] = np.random.exponential(scale=10, size=num_impulses)

#h = np.array([0.5, 1.0, -1.0, 2.5, -1.0, 1.0, 0.5])
#h = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
h = np.array([0.1, 0.15, 0.4, 0.5, 0.4, 0.15, 0.1])

y = np.convolve(x, h) + 0.1*np.random.randn(len(x) + len(h) - 1)


m = len(y) - len(x) + 1
zpad = np.zeros(m-1)
xp = np.r_[x, zpad]


#------------------------------------------------------------------

#h_deconv, h_r = deconvolve(y, x)

#------------------------------------------------------------------
