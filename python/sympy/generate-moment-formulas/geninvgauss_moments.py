
import sympy
from sympy import besselk, sqrt, pprint, simplify
from genmom import generate_moments

#
# Notation is from the Wikipedia page
#     https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution
#
a, b, p, t = sympy.symbols('a,b,p,t')
mgf = (a / (a - 2*t))**(p/2) * (besselk(p, sqrt(b*(a - 2*t)))
                                / besselk(p, sqrt(a*b)))
moments = generate_moments(mgf, t, 2)
print()
print("generalized inverse gaussian distribution moments about 0")
print("(maybe could be simplified by using the recurrence relations for K_p(z))")
for k, m in enumerate(moments):
    print(f'\nm{k+1}:')
    pprint(simplify(m))
