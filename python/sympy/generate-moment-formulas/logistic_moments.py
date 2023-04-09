
import sympy
from sympy import exp, beta
from genmom import generate_moments

#
# Notation is from the Wikipedia page
#     https://en.wikipedia.org/wiki/Logistic_distribution
#
mu, s, t = sympy.symbols('mu,s,t')
mgf = exp(mu*t)*beta(1 - s*t, 1 + s*t)
moments = generate_moments(mgf, t, 3)
print()
print("logistic distribution moments about 0")
for k, m in enumerate(moments):
    print(f'm{k+1}: {m}')
