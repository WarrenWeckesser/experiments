from mpmath import mp
import numpy as np
from scipy.fft import dct
import matplotlib.pyplot as plt
from chebstuff import chebyshev_coefficients, chebyshev_eval, chebyshev_lobatto_sample

mp.dps = 250


def foo(x):
    return np.exp(-x) * np.sin(x)


def foo_mp(x):
    return mp.exp(-x) * mp.sin(x)


# For the problem as currently implemented, increasing N beyond 17 doesn't
# decrease the max_relerr computed below.
N = 17
a = 1
b = 3


t, x, y = chebyshev_lobatto_sample(mp, foo_mp, a, b, N)


c = dct(y, type=1)
c /= N
c[0] /= 2
c[-1] /= 2

cheb = np.polynomial.Chebyshev.fit(np.array(x).astype(float),
                                   np.array(y).astype(float),
                                   domain=[a, b], deg=N)


mp_coefs = chebyshev_coefficients(mp, y, N)

# c, cheb.coef and mp_coefs should be the same (except for floating point precision
# differences).
