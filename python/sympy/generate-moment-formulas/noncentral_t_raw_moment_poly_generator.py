from functools import lru_cache
from mpmath import mp


@lru_cache
def _poly_coeffs(k):
    """
    Generate the coefficients of the polynomial that is the
    result of the expression exp(-x**2/2) * (d^k/dx^k)exp(x**2/2).
    """
    if k == 0:
        return [1]
    c = [0, 1]
    if k == 1:
        return c
    for _ in range(2, k+1):
        c = [a + b for a, b in zip([i*j for i, j in enumerate(c[1:], start=1)],
                                   [0] + c[:-2])]
        c.extend([0, 1])
    return c


def nct_noncentral_moment(k, nu, mu):
    """
    Parameters
    ----------
    nu : degrees of freedom
    mu : noncentrality parameter
    """
    with mp.extradps(5):
        nu = mp.mpmathify(nu)
        mu = mp.mpmathify(mu)
        if nu <= k:
            return mp.nan
        c = _poly_coeffs(k)
        return mp.exp((k/2)*mp.log(nu/2) + mp.loggamma((nu-k)/2) - mp.loggamma(nu/2))*mp.polyval(c[::-1], mu)
