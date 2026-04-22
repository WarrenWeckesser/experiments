import sympy
from sympy import init_printing
import numpy as np
import matplotlib.pyplot as plt


init_printing()

# @mp.extradps(5)
# def pdf(x, mu, sigma):
#     """
#     Probability density function of the folded normal distribution.
#     """
#     x = mp.mpf(x)
#     mu = mp.mpf(mu)
#     sigma = _validate_sigma(sigma)
#     if x < 0:
#         return mp.zero
#     return mp.npdf(x, mu, sigma) + mp.npdf(-x, mu, sigma)

x, mu, sigma, c = sympy.symbols('x mu sigma c')
# PDF:
z1 = (x - mu) / sigma
z2 = (-x - mu) / sigma
f = (sympy.exp(-z1**2/2) + sympy.exp(-z2**2/2)) / (c * sigma)

df = sympy.diff(f, x)
df = sympy.simplify(df)

print("Derivative of the PDF of the folded normal distribution:")
sympy.pprint(df)

df2 = sympy.diff(df, x)
df2 =  sympy.simplify(df2)

print()
print("Second derivative of the PDF of the folded normal distribution:")
sympy.pprint(df2)

print()
t = df2.subs(x, 0)
t = sympy.simplify(t)
sympy.pprint(t)

tmu1 = t.subs(mu, sigma)
tmu1 = sympy.simplify(tmu1)
sympy.pprint(tmu1)
