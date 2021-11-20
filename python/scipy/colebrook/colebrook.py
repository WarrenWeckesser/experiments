
import math
import numpy as np
from scipy.special import lambertw


def colebrook(Re, Dh, eps):
    """
    Solve the Colebrook equation for f, given Re, D and eps.

    The equation is also known as the Colebrook-White equation.

    Quoting from the wikipedia article [1]_,

        The phenomenological Colebrook–White equation (or Colebrook
        equation) expresses the Darcy friction factor f as a function
        of Reynolds number Re and pipe relative roughness ε / Dh,
        fitting the data of experimental studies of turbulent flow in
        smooth and rough pipes.  The equation can be used to (iteratively)
        solve for the Darcy–Weisbach friction factor f.

        For a conduit flowing completely full of fluid at Reynolds numbers
        greater than 4000, it is expressed as:

            1/sqrt(f) = -2*log(eps/(3.7*Dh) + 2.51/(Re*sqrt(f))

    Parameters
    ----------
    Re : array_like
        Reynolds number
    Dh : array_like
        Hydraulic diameter.  From wikipedia: "For fluid-filled circular
        conduits, Dh = D = inside diameter."  Must have the same units as eps.
    eps : array_like
        Absolute pipe roughness.  Must have the same units as Dh.

    Returns
    -------
    f : ndarray
        The Darcy friction factor.

    References
    ----------
    .. [1] "Darcy friction factor formulae, Wikipedia,
           https://en.wikipedia.org/wiki/Darcy_friction_factor_formulae#Colebrook%E2%80%93White_equation

    Examples
    --------
    >>> colebrook(Re=5e6, Dh=1250.0, eps=[0.01, 0.02, 0.04, 0.08])
    array([0.00947893, 0.00988838, 0.01054755, 0.0115219 ])

    """
    Re = np.asarray(Re)
    Dh = np.asarray(Dh)
    eps = np.asarray(eps)
    a = 2.51 / Re
    b = eps / (3.7*Dh)
    p = 1/math.sqrt(10)
    lnp = math.log(p)
    x = -lambertw(-lnp/a * np.power(p, -b/a))/lnp - b/a
    if np.any(x.imag != 0):
        raise ValueError('x is complex')
    f = 1/x.real**2
    return f
