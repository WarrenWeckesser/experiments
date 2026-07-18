
import sympy
from sympy import exp, gamma
from sympy.polys.polyfuncs import horner
from genmom import generate_moments

#
# Notation is from the Wikipedia page
#     https://en.wikipedia.org/wiki/Gumbel_distribution
#
mu, beta, t = sympy.symbols('mu,beta,t')
mgf = exp(mu*t)*gamma(1 - beta*t)
moments = generate_moments(mgf, t, 5)
print()
print("Gumbel distribution moments about 0")
for k, m in enumerate(moments):
    print(f'\nm{k+1}:')
    sympy.pprint(m)
    mh_beta = horner(m, wrt=beta)
    sympy.pprint(mh_beta)
    mh_both = horner(m, wrt=(mu, beta))
    sympy.pprint(mh_both)
