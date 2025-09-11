"""
Generate a couple plots to illustrate the functions used in the
rejection method for generating Zipfian random variates.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import boxcox


def target(x, a, n):
    # This is the target "histogram function" i.e. the nonnormalized PMF,
    # expanded to be a function of the continuous variable x.
    mask = (x > 0.5) & (x < n + 0.5)
    out = np.zeros_like(x)
    out[mask] = np.round(x[mask])**-a
    return out


def g(x, a, n):
    # g is the "hat" function
    out = np.zeros(len(x))
    out[(0.5 <= x) & (x <= 1.5)] = 1.0
    mask = (1.5 < x) & (x < n + 0.5)
    out[mask] = (x[mask] - 0.5)**-a
    return out


def G(x, a, n):
    # Currently, the only call of this function used in the
    # rejection method is G(n + 0.5, a, n).
    x = np.atleast_1d(x)
    out = np.zeros(len(x))
    out[x >= n + 0.5] = boxcox(n, 1 - a) + 1
    mask1 = (0.5 <= x) & (x <= 1.5)
    out[mask1] = x[mask1] - 0.5
    mask = (1.5 < x) & (x < n + 0.5)
    out[mask] = boxcox(x[mask] - 0.5, 1 - a) + 1
    return out


a = 0.95
n = 7

x1 = np.arange(1.0, n + 1)
xx = np.linspace(0.0, n + 1.0, 8000)
y_target = target(xx, a, n)
y_hatfunc = g(xx, a, n)

figsize = (7.5, 5.25)

# Figure 1.

plt.figure(figsize=figsize)

plt.plot(x1, target(x1, a, n), 'o', ms=3.5,
         label='Nonnormalized Zipfian PMF, '
               f'$k^{{-a}}$ for $k\\in\\{{1, 2, ..., {n}\\}}$')
plt.plot(xx, y_target,
         label='target(x, a, n): target histogram function\n'
               '(nonnormalized piecewise constant PDF\n'
               'associated with the Zipfian PMF)')
plt.plot(xx, y_hatfunc, 'k--', alpha=0.6,
        label='g(x, a, n): hat function\n'
              '(nonnormalized PDF of the dominating distribution)')

plt.legend(shadow=True, framealpha=1)
plt.grid(visible=True)
plt.xlabel('x')
plt.title(f'Zipfian Rejection Method (a={a}, n={n})')

# Figure 2.

plt.figure(figsize=figsize)

plt.plot(xx, G(xx, a, n), 'k',
         label='G(x, a, n)\nintegral of the hat function g(x, a, n)')
plt.plot(0.5, 0, 'k.')
plt.plot(1.5, G(1.5, a, n), 'k.')
maxG = G(n + 0.5, a, n)
plt.plot(n + 0.5, maxG, 'k.')
plt.axhline(maxG, linestyle=':', alpha=0.5,
            label='max G(x, a, n) = G(n + 0.5, a, n)')

plt.legend(shadow=True, framealpha=1)
plt.grid(visible=True)
plt.xlabel('x')
plt.title(f'Zipfian Rejection Method  (a={a}, n={n})\n'
          'G(x, a, n), the nonnormalized CDF of the dominating distribution')

plt.show()
