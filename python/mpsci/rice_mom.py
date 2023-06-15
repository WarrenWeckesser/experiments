from mpmath import mp
from mpsci.stats import mean as _mean
from mpsci.distributions._common import _validate_x_bounds


def rice_mom(x):
    """
    Method of moments parameter estimation for the Rice distribution.

    x must be a sequence of numbers.

    Returns (nu, sigma).

    Note: This implementation is experimental.  It is not unusual for
    the numerical solver to fail.
    """
    with mp.extradps(5):
        x = _validate_x_bounds(x, low=0, strict_low=True)
        M1 = _mean(x)
        M2 = _mean([mp.mpf(t)**2 for t in x])
        nhalf = -mp.one/2
        c = mp.sqrt(mp.pi/2)

        def func(s):
            ss = 2*s**2
            return c*s*mp.hyp1f1(nhalf, 1, (ss - M2)/ss) - M1

        f0 = mp.limit(func, 0)
        t0 = mp.mpf(2e-16)
        t = t0
        f1 = func(t)
        neg = []
        tmin = t0
        fmin = f1
        while f1 <= f0:
            t = 1.6*t
            f1 = func(t)
            # print(t, f1)
            if f1 < 0:
                neg.append(t)
            if f1 < fmin:
                tmin = t
                fmin = f1
        if len(neg) > 0:
            interval = (t0, neg[0])
            sigma = mp.findroot(func, interval, method='bisect')
        else:
            sigma = mp.findroot(func, tmin)
        nu = mp.sqrt(M2 - 2*sigma**2)
        return nu, sigma
