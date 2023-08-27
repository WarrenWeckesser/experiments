import numpy as np
from scipy.stats import levy_stable


_PI_2 = np.pi/2

# _LANDAU_XSTAR was found numerically.  It is the mode of the "standard"
# Landau distribution.
_LANDAU_XSTAR = -0.2227829812564085


def landau_gen(mpv=None, scale=1):
    """
    Create a Landau distribution with its mode at mpv (the "most
    probable value") and with the given scale.  If mpv is not given,
    the default distribution is the one that is often called *the*
    Landau distribution; it corresponds to mpv approximately
    -0.2227829812564085.

    The return value is a SciPy "frozen" distribution, with methods
    `pdf(x)`, `cdf(x)`,  etc.

    Do not use the rvs method of the distribution that is returned
    unless you are using SciPy 1.7.0 or newer.  There is a bug in
    old versions of the SciPy implementation of the rvs method
    of levy_stable for the case alpha=1.
    """
    # The Landau distribution is a special case of the Levy stable
    # family of distributions.  loc and scale are computed so that
    # using scale=1 in the returned distribution corresponds to the
    # scale of the "standard" Landau distribution (e.g. as in the
    # wikipedia article on the distribution).
    loc_star = scale*np.log(_PI_2)
    if mpv is not None:
        loc_star += -scale*_LANDAU_XSTAR + mpv
    return levy_stable(1, 1, loc=loc_star, scale=_PI_2*scale)
