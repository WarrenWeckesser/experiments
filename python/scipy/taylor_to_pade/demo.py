import numpy as np
import matplotlib.pyplot as plt
from taylor_to_pade import taylor_to_pade


# From the taylor_to_pade docstring, but with fewer Taylor coefficients.

# Coefficients of the Taylor polynomial approximation to the function
# log(1 + x) at x = 0:
c = [0, 1, -1/2, 1/3, -1/4]

# Compute the coefficients of the Padé approximant:
cp, cq = taylor_to_pade(c)

T = np.polynomial.Polynomial(c)
P = np.polynomial.Polynomial(cp)
Q = np.polynomial.Polynomial(cq)

x = np.linspace(-0.975, 5, 2000)

plt.plot(x, np.log1p(x), 'k', linewidth=1, label='log(1+x)')
plt.plot(x, T(x), 'r--', alpha=0.4, label='Taylor T(x)')
plt.plot(x, P(x)/Q(x), 'g', alpha=0.4, linewidth=2.5, label='Padé P(x)/Q(x)')
plt.plot(0, 0, 'k.')
plt.ylim(-4, 2)
plt.xlabel('x')
plt.legend(framealpha=1, shadow=True)
plt.grid()

# plt.show()
plt.savefig('demo.svg')
