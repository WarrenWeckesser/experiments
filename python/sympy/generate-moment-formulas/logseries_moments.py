
import sympy
from sympy import exp, log
from genmom import generate_moments

#
# Notation and MGF is from the Wikipedia page
#     https://en.wikipedia.org/wiki/Logarithmic_distribution
#
p, t = sympy.symbols('p,t')
# This leaves out the denominator log(1 - p), since it is independent
# of t.  The expressions for the moment must be divided by log(1 - p)
# later.
mgf_numer = log(1 - p*exp(t))
moments = generate_moments(mgf_numer, t, 8)
print()
print("logistic distribution moments about 0")
for k, m in enumerate(moments):
    print(f'm{k+1}: {m}')
