# Copyright (c) 2022, Warren Weckesser
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import math
import numpy as np
from scipy.special import lambertw


def colebrook(Re, Dh, eps, constant1=3.71):
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

    Many references use the constant 3.71 instead of 3.7, and this
    function does, too.  If one needs to used 3.7, specify that value as
    the parameter `constant1`.

    Parameters
    ----------
    Re : array_like
        Reynolds number
    Dh : array_like
        Hydraulic diameter.  From wikipedia: "For fluid-filled circular
        conduits, Dh = D = inside diameter."  Must have the same units as eps.
    eps : array_like
        Absolute pipe roughness.  Must have the same units as Dh.
    constant1 : float, optional
        This option allows the constant 3.71 in the implicit equation
        to be replaced with some other values (e.g. 3.7).

    Returns
    -------
    f : ndarray
        The Darcy friction factor.

    References
    ----------
    .. [1] "Darcy friction factor formulae", Wikipedia,
           https://en.wikipedia.org/wiki/Darcy_friction_factor_formulae#Colebrook%E2%80%93White_equation

    Examples
    --------
    >>> colebrook(Re=5e6, Dh=1250.0, eps=[0.01, 0.02, 0.04, 0.08])
    array([0.00947772, 0.00988635, 0.01054441, 0.01151743])

    """
    Re = np.asarray(Re)
    Dh = np.asarray(Dh)
    eps = np.asarray(eps)
    a = 2.51 / Re
    b = eps / (constant1*Dh)
    p = 1/math.sqrt(10)
    lnp = math.log(p)
    x = -lambertw(-lnp/a * np.power(p, -b/a))/lnp - b/a
    if np.any(x.imag != 0):
        raise ValueError('x is complex')
    f = 1/x.real**2
    return f
