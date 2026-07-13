from dataclasses import dataclass, field
import numpy as np
from scipy import special
from scipy.integrate import quad

"""
Notes

                   Domain    w is 1  #params
roots_chebyc       -2, 2               0 
roots_chebys       -2, 2               0
roots_chebyt       -1, 1               0
roots_chebyu       -1, 1               0
roots_gegenbauer   -1, 1               1
roots_genlaguerre   0, inf             1
roots_hermite      -inf, inf           0
roots_hermitenorm  -inf, inf           0
roots_jacobi       -1, 1               2
roots_laguerre      0, inf             0
roots_legendre     -1, 1       1       0
roots_sh_chebyt     0, 1               0
roots_sh_chebyu     0, 1               0
roots_sh_jacobi     0, 1               2
roots_sh_legendre   0, 1       1       0


All the methods are exact up to degree 2*n-1.
"""


@dataclass
class Func:
    funcname: str
    wmath: str
    wcode: str
    a: float
    b: float
    params: list[str] = field(default_factory=list)
    param_values: list[float] = field(default_factory=list)


template = """
Examples
--------
>>> import numpy as np
>>> from scipy.special import {funcname}
>>> from scipy.integrate import quad

Take a look at the points and weights for ``n == 3``.
{assign_params2}
>>> n = 3
>>> x, weights = {funcname}(n{args})
>>> x
{x3!r}
>>> wts
{wts3!r}

If we include the parameter ``mu=True``, the sum of the weights is also
returned.

>>> x, weights, mu = {funcname}(n{args}, mu=True)
>>> mu, weights.sum()
({mu3!r}, {sum3!r})

All the Gauss quadrature rules define a method for approximating an integral
of the form  int_a^b f(x)w(x)dx over a certain interval [a, b] for a certain
weight function w(x).

For {propername} quadrature,

* [a, b] = [{a}, {b}]
* w(x{args}) = {wmath}

Here we compare the quadrature formula to the result computed with
`scipy.integrate.quad` for the integrand `(1 + 3*x - 2*x**2 + x**4)*w(x{args})`,
so f(x) is a polynomial with degree 4.

The polynomial f(x) and the integrand of our desired integral:

>>> f = np.polynomial.Polynomial([1, 3, -2, 0, 1])
>>> def integrand(x{args}):
...     return {integrandcode}

Compute the integral using `scipy.integrate.quad`:

>>> intgrl, err = quad(integrand, {a}, {b}, epsrel=1e-12{quadargs})
>>> intgrl
{q!r}

Compute the integral using the quadrature rule.  We'll use n=3 (coefficients
computed above), so the quadrature is exact for this function, and we expect
good agreement with the integral computed with `scipy.integerate.quad`.

>>> weighted_sum = weights @ f(x)
{weighted_sum!r}

As expected, we have good agreement between ``intgrl`` and ``weighted_sum``.
"""

# This must match the code in the template above.
f = np.polynomial.Polynomial([1, 3, -2, 0, 1])

funcs = [
    Func(funcname='roots_chebyc',
         wmath='1 / sqrt(1 - (x/2)^2)',
         wcode='1/np.sqrt((1 - x/2) * (1 + x/2))',
         a=-2,
         b=2),
    Func(funcname='roots_chebyu',
         wmath='sqrt(1 - x^2)',
         wcode='np.sqrt((1 - x) * (1 + x))',
         a=-1,
         b=1),
    Func(funcname='roots_gegenbauer',
         params=['alpha'],
         param_values=[1.25],
         wmath='(1 - x^2)^(alpha - 1/2)',
         wcode='(1 - x**2)**(alpha - 0.5)',
         a=-1,
         b=1),
    Func(funcname='roots_hermite',
         wmath='exp(-x^2)',
         wcode='np.exp(-x**2)',
         a=-np.inf,
         b=np.inf),
    Func(funcname='roots_jacobi',
         params=['alpha', 'beta'],
         param_values=[0.75, 1.25],
         wmath='(1 - x)^alpha + (1 + x)^beta',
         wcode='(1 - x)**alpha * (1 + x)**beta',
         a=-1,
         b=1),
    Func(funcname='roots_legendre',
         wmath='1',
         wcode='1',
         a=-1,
         b=1),
    Func(funcname='roots_sh_chebyt',
         wmath='1/sqrt(x * (1 - x))',
         wcode='1/np.sqrt(x * (1 - x))',
         a=0,
         b=1),
    Func(funcname='roots_sh_jacobi',
         params=['p1', 'q1'],
         param_values=[0.75, 1.25],
         wmath='(1 - x)^(p1 - q1) * x^(q1 - 1)',
         wcode='(1 - x)**(p1 - q1) * x**(q1 - 1)',
         a=0,
         b=1),
]

quadstr = """
from math import inf

def integrand(x{args}):
    return f(x) * {wcode}

q, err = quad(integrand, {a}, {b}, epsrel=1e-12{quadargs})
"""

for func in funcs:
    args = ''.join([', ' + param for param in func.params])
    if len(func.params) > 0:
        quadargs = ', args=(' + ', '.join(func.params) + ')'
    else:
        quadargs = ''

    assign_params = '\n'.join([f"{name} = {value}" for name, value in zip(func.params, func.param_values)])
    code = assign_params + quadstr.format(args=args, wcode=func.wcode, a=func.a, b=func.b, quadargs=quadargs)
    exec(code, globals=globals())

    n = 3
    roots_func = getattr(special, func.funcname)
    propername = roots_func.__doc__.split('\n')[0].replace(' quadrature.', '')
    x, wts, mu = roots_func(n, *func.param_values, mu=True)
    sum3 = wts.sum()
    weighted_sum = wts @ f(x)

    if not np.allclose(weighted_sum, q, rtol=err/q):
        raise RuntimeError(f"values don't match for {func.funcname}")

    assign_params2 = ''.join([f"\n>>> {name} = {value}" for name, value in zip(func.params, func.param_values)])

    #wfactor = "*" + func.wcode if func.wcode != "1" else ""
    if func.wcode == "1":
        integrandcode = f"f(x)"
    elif func.wcode.startswith('1/'):
        integrandcode = f"f(x) / {func.wcode[2:]}"
    else:
        integrandcode = f"f(x) * {func.wcode}"

    text = template.format(funcname=func.funcname, propername=propername,
                           x3=x, wts3=wts, mu3=mu, sum3=sum3,
                           wmath=func.wmath, integrandcode=integrandcode, a=func.a, b=func.b,
                           assign_params2=assign_params2,
                           args=args, quadargs=quadargs, q=q, weighted_sum=weighted_sum)
    print("-"*52)
    print(f"*** {func.funcname} ***")
    print("-"*52)
    print(text)
