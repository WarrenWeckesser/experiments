import numpy as np
from scipy.linalg import toeplitz, solve


def taylor_to_pade(c):
    """
    Convert the coefficients of a Taylor polynomial to the coefficients
    of a Padé approximant.

    This is not the most efficient algorithm, and it might fail
    in some cases.

    Parameters
    ----------
    c : 1-d sequence
        Taylor coefficients; c[k] is the coefficient of x^k.
        len(a) must be odd.

    Return Value
    ------------
    cp, cq : ndarray
        Coefficents of numerator and denominator of the Pade
        approximant p(x)/q(x), respectively.

    Examples
    --------
    Coefficients of the Taylor polynomial approximation to the function
    log(1 + x) at x = 0:

    >>> c = [0, 1, -1/2, 1/3, -1/4, 1/5, -1/6, 1/7, -1/8, 1/9, -1/10]

    Compute the coefficients of the Padé approximant:

    >>> cp, cq = taylor_to_pade(c)
    >>> cp  # numerator
    array([-0.        ,  1.        ,  2.        ,  1.30555556,  0.30555556,
            0.01812169])
    >>> cq  # denominator
    array([1.        , 2.5       , 2.22222222, 0.83333333, 0.11904762,
           0.00396825])
    """
    c = np.asarray(c)
    n = len(c)
    m, rem = divmod(n, 2)
    if rem != 1:
        raise ValueError("len(a) must be odd")

    M = np.zeros((n, n))
    M[:m, :m] = toeplitz(c[m:0:-1], c[m:-1])
    M[m:, :m] = toeplitz(np.r_[c[0], np.zeros(m)], c[:m])
    M[m:, m:] = -np.eye(m + 1)

    x = solve(M, -c[::-1])
    v = x[m-1::-1]
    u = x[:m-1:-1]
    return u, np.r_[1, v]
