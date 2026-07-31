from mpmath import mp
from mpsci.fun import digammainv as mp_digammainv
import numpy as np
from scipy.special import digamma, digammainv
from scipy import optimize
import matplotlib.pyplot as plt


def _digammainv(y):
    """Inverse of the digamma function (real positive arguments only).

    This function is used in the `fit` method of `gamma_gen`.
    The function uses either optimize.fsolve or optimize.newton
    to solve `sc.digamma(x) - y = 0`.  There is probably room for
    improvement, but currently it works over a wide range of y:

    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> y = 64*rng.standard_normal(1000000)
    >>> y.min(), y.max()
    (-311.43592651416662, 351.77388222276869)
    >>> x = [_digammainv(t) for t in y]
    >>> np.abs(sc.digamma(x) - y).max()
    1.1368683772161603e-13

    """
    _em = 0.5772156649015328606065120

    def func(x):
        return digamma(x) - y

    if y > -0.125:
        x0 = np.exp(y) + 0.5
        if y < 10:
            # Some experimentation shows that newton reliably converges
            # much faster than fsolve in this y range.  For larger y,
            # newton sometimes fails to converge.
            value = optimize.newton(func, x0, tol=1e-10)
            return value
    elif y > -3:
        x0 = np.exp(y/2.332) + 0.08661
    else:
        x0 = 1.0 / (-y - _em)

    value, info, ier, mesg = optimize.fsolve(func, x0, xtol=1e-11,
                                             full_output=True)
    if ier != 1:
        raise RuntimeError(f"_digammainv: fsolve failed, y = {y!r}")

    return value[0]



mp.dps = 75
xx1 = np.linspace(-500, 500, 5000)
xx2 = np.linspace(0, 40, 2500)
xx = np.concatenate((xx1, xx2))
# rng = np.random.default_rng(121263137472525314065)
# xx = rng.normal(scale=50, size=5000)
y_old = np.array([_digammainv(t) for t in xx])
y_new = digammainv(xx)
ref = np.array([float(mp_digammainv(t)) for t in xx])
re_old = abs(y_old - ref) / ref
re_new = abs(y_new - ref) / ref

plt.plot(xx, re_old, 'k.', label='old')
plt.plot(xx, re_new, 'co', alpha=0.5, label='new')
plt.grid(True, alpha=0.5)
plt.legend(shadow=True)
# plt.semilogy()
plt.show()
