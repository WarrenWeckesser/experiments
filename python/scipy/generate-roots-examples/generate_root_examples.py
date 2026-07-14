from dataclasses import dataclass, field
import numpy as np
import re
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
    wsum: str = ''
    params: list[str] = field(default_factory=list)
    param_values: list[float] = field(default_factory=list)


def get_docstring(code, funcname):
    def_start = code.find(f"def {funcname}(")
    docstring_start = code[def_start:].find('"""') + 3
    start = def_start + docstring_start
    docstring_len = code[start:].find('"""')
    return start, docstring_len


with open('_orthogonal.py', 'r') as f:
    original_code = f.read()


template1 = """
Examples
--------
>>> import numpy as np
>>> from scipy.special import {funcname}, {evalname}
>>> from scipy.integrate import quad

Take a look at the sample points and weights for ``n = {n}``.
{assign_params2}
>>> n = {n}
>>> x, weights = {funcname}(n{args})
>>> x
{x3!r}
>>> wts
{wts3!r}

Verify that ``x`` are the roots of the degree-{n} polynomial.
These values might not be *exactly* 0 because of floating point imprecision:

>>> {evalname}({n}{args}, x)
{evalresult!r}

If we include the parameter ``mu=True``, the sum of the weights is also
returned.{sumcomment}

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
`scipy.integrate.quad` for the integrand f(x)*w(x{args}), where f(x)
is this polynomial with degree 5:

    f(x) = 1 + 3*x - 2*x**2 - x**4 + x**5/10
"""

templatew1 = """
Define the polynomial f(x):

>>> f = np.polynomial.Polynomial([1, 3, -2, 0, -1, 0.1])

Compute the integral using `scipy.integrate.quad`:

>>> intgrl, err = quad(f, {a}, {b}, epsrel=1e-12{quadargs})
>>> intgrl
{q!r}
"""

templatewx = """
Define the polynomial f(x) and the integrand of our desired integral:

>>> f = np.polynomial.Polynomial([1, 3, -2, 0, -1, 0.1])
>>> def integrand(x{args}):
...     return {integrandcode}

Compute the integral using `scipy.integrate.quad`:

>>> intgrl, err = quad(integrand, {a}, {b}, epsrel=1e-12{quadargs})
>>> intgrl
{q!r}
"""

template2 = """
Compute the integral using the quadrature rule.  We'll use the sample
points and weights computed above with n = {n}.  Since the degree of
the polynomial f(x) is less than 2*n, the quadrature is exact for this
function, and we expect good agreement with the integral computed with
`scipy.integrate.quad`.

>>> weighted_sum = weights @ f(x)
{weighted_sum!r}

As expected, we have good agreement between ``intgrl`` and ``weighted_sum``.
"""

# This must match the code in the template above.
f = np.polynomial.Polynomial([1, 3, -2, 0, -1, 0.1])

funcs = [
    Func(funcname='roots_chebyc',
         wmath='1 / sqrt(1 - (x/2)^2)',
         wcode='1/np.sqrt((1 - x/2) * (1 + x/2))',
         wsum='2*pi',
         a=-2,
         b=2),
    Func(funcname='roots_chebys',
         wmath='sqrt(1 - (x/2)^2)',
         wcode='np.sqrt((1 - x/2) * (1 + x/2))',
         wsum='pi',
         a=-2,
         b=2),
    Func(funcname='roots_chebyt',
         wmath='1/sqrt(1 - x^2)',
         wcode='1/np.sqrt((1 - x) * (1 + x))',
         wsum='pi',
         a=-1,
         b=1),
    Func(funcname='roots_chebyu',
         wmath='sqrt(1 - x^2)',
         wcode='np.sqrt((1 - x) * (1 + x))',
         wsum='pi/2',
         a=-1,
         b=1),
    Func(funcname='roots_gegenbauer',
         params=['alpha'],
         param_values=[1.25],
         wmath='(1 - x^2)^(alpha - 1/2)',
         wcode='((1 - x) * (1 + x))**(alpha - 0.5)',
         a=-1,
         b=1),
    Func(funcname='roots_genlaguerre',
         params=['alpha'],
         param_values=[1.25],
         wmath='x^alpha * exp(-x)',
         wcode='x**alpha * np.exp(-x)',
         a=0,
         b=np.inf),
    Func(funcname='roots_hermite',
         wmath='exp(-x^2)',
         wcode='np.exp(-x**2)',
         wsum='sqrt(pi)',
         a=-np.inf,
         b=np.inf),
    Func(funcname='roots_hermitenorm',
         wmath='exp(-x^2/2)',
         wcode='np.exp(-x**2/2)',
         wsum='sqrt(2*pi)',
         a=-np.inf,
         b=np.inf),
    Func(funcname='roots_jacobi',
         params=['alpha', 'beta'],
         param_values=[0.75, 1.25],
         wmath='(1 - x)^alpha + (1 + x)^beta',
         wcode='(1 - x)**alpha * (1 + x)**beta',
         a=-1,
         b=1),
    Func(funcname='roots_laguerre',
         wmath='exp(-x)',
         wcode='np.exp(-x)',
         wsum='1',
         a=0,
         b=np.inf),
    Func(funcname='roots_legendre',
         wmath='1',
         wcode='1',
         wsum='2',
         a=-1,
         b=1),
    Func(funcname='roots_sh_chebyt',
         wmath='1/sqrt(x * (1 - x))',
         wcode='1/np.sqrt(x * (1 - x))',
         wsum='pi',
         a=0,
         b=1),
    Func(funcname='roots_sh_chebyu',
         wmath='sqrt(x - x^2))',
         wcode='np.sqrt(x * (1 - x))',
         a=0,
         b=1),
    Func(funcname='roots_sh_jacobi',
         params=['p1', 'q1'],
         param_values=[0.75, 1.25],
         wmath='(1 - x)^(p1 - q1) * x^(q1 - 1)',
         wcode='(1 - x)**(p1 - q1) * x**(q1 - 1)',
         a=0,
         b=1),
    Func(funcname='roots_sh_legendre',
         wmath='1',
         wcode='1',
         wsum='1',
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
        # quadargs = ', args=(' + ', '.join(func.params) + ')'
        quadargs = ', args=(' + ' '.join(parname + ',' for parname in func.params) + ')'
    else:
        quadargs = ''

    assign_params = '\n'.join([f"{name} = {value}" for name, value in zip(func.params, func.param_values)])
    code = assign_params + quadstr.format(args=args, wcode=func.wcode, a=func.a, b=func.b, quadargs=quadargs)
    exec(code, globals=globals())

    n = 4

    evalname = "eval_" + func.funcname.split('_', 1)[1]

    roots_func = getattr(special, func.funcname)
    eval_func = getattr(special, evalname)
    propername = roots_func.__doc__.split('\n')[0].replace(' quadrature.', '')
    x, wts, mu = roots_func(n, *func.param_values, mu=True)
    evalresult = eval_func(n, *func.param_values, x)
    sum3 = wts.sum()
    weighted_sum = wts @ f(x)

    if not np.allclose(weighted_sum, q, rtol=np.abs(err/q)):
        raise RuntimeError(f"values don't match for {func.funcname}: {weighted_sum = }  {q = }  {err = }")

    assign_params2 = ''.join([f"\n>>> {name} = {value}" for name, value in zip(func.params, func.param_values)])

    evalname = "eval_" + func.funcname.split('_', 1)[1]

    #wfactor = "*" + func.wcode if func.wcode != "1" else ""
    if func.wcode == "1":
        integrandcode = f"f(x)"
    elif func.wcode.startswith('1/'):
        integrandcode = f"f(x) / {func.wcode[2:]}"
    else:
        integrandcode = f"f(x) * {func.wcode}"

    if func.wcode == "1":
        template = ''.join([template1, templatew1, template2])
    else:
        template = ''.join([template1, templatewx, template2])

    sumcomment = ''
    if func.wsum != '':
        sumcomment = f" The sum of the weights for {propername} quadrature\nis always {func.wsum}."

    text = template.format(funcname=func.funcname, propername=propername,
                           n=n, x3=x, wts3=wts, mu3=mu, sum3=sum3,
                           wmath=func.wmath, integrandcode=integrandcode, a=func.a, b=func.b,
                           assign_params2=assign_params2, evalname=evalname, evalresult=evalresult,
                           args=args, quadargs=quadargs, q=q, weighted_sum=weighted_sum,
                           sumcomment=sumcomment)
    print("-"*52)
    print(f"*** {func.funcname} ***")
    print("-"*52)
    print(text)
