
import sympy
from sympy import exp, expint
from sympy.polys.polyfuncs import horner
from genmom import generate_moments

#
# Notation is from the Wikipedia page
#     https://en.wikipedia.org/wiki/Gompertz_distribution
#
eta, t = sympy.symbols('eta,t')
# Note that with the first argument of expint being t, there is an alternating
# sign of the output formulas.  Is this expected?  Changing t to -t removes
# the sign alternation.
mgf = eta * exp(eta) * expint(t, eta)
moments = generate_moments(mgf, t, 5, direct_eval_limit=True)
moments = [horner(m, wrt=eta) for m in moments]
print()
print("Gompertz distribution moments about 0")
for k, m in enumerate(moments):
    print(f'\nm{k+1}:')
    sympy.pprint(m)
