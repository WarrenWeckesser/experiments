
import mpmath
import numpy as np


def pow1pxm1_naive(x, a):
    return (1 + x)**a - 1


def pow1pxm1_ts(x, a):
    if abs(x) < 0.002:
        c1 = a
        c2 = a*(-0.5 + 0.5*a)
        c3 = a*(1/3 + a*(-0.5 + a/6))
        c4 = a*(-0.25 + a*(11/24 + a*(-0.25 + a/24)))
        c5 = a*(0.2 + a*(-5/12 + a*(7/24 + a*(-1/12 + a/120))))
        return x*(c1 + x*(c2 + x*(c3 + x*(c4 + x*c5))))
    else:
        return pow1pxm1_naive(x, a)


def pow1pxm1(x, a):
    """
    Compute (1 + x)**a - 1
    """
    fp = a
    fp2 = a*(a-1)
    fp3 = a*(a-1)*(a-2)

    x = np.atleast_1d(x)
    eps = np.finfo(np.float64).eps
    big = np.abs(x) > eps**(1/4)
    print(np.count_nonzero(big))
    bigvals = x[big]
    notbigvals = x[~big]
    y = np.empty(x.shape, dtype=np.float64)
    y[big] = (1 + bigvals)**a - 1
    y[~big] = notbigvals*(fp + fp2*notbigvals/2 + fp3*notbigvals**2/6)
    return y


def pow1pxm1_naive_mp(x, a):
    """
    Compute (1 + x)**a - 1
    """

    with mpmath.extradps(15):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        return (1 + x)**a - 1


"""

if __name__ == "__main__":
    mpmath.mp.dps = 15
    a = mpmath.mpf('2.75')
    x = mpmath.linspace(-0.1, 0.1, 50001)
    yn = [pow1pxm1_naive(t, a) for t in x]
    yg = [pow1pxm1_mp(t, a) for t in x]
    with mpmath.workdps(60):
        yh = [pow1pxm1_naive(t, a) for t in x]

        maxerr = max([abs(a - b)/abs(b) for (a, b) in zip(yn, yh)])
        print(maxerr)

        maxerr = max([abs(a - b)/abs(b) for (a, b) in zip(yg, yh)])
        print(maxerr)
"""
