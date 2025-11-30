import numpy as np
from scipy.stats import nhypergeom, betabinom
import matplotlib.pyplot as plt


M = 25
n = 10
r = 5

k = np.arange(0, n + 1)

nh = nhypergeom(M, n, r)
print("nh:", nh.mean(), nh.var())
pmf_nh = nh.pmf(k)

# Convert nhypergeom parameters to betabinom parameters.
a = r
b = M - n - r + 1
# n = n

bb = betabinom(n, a, b)
print("bb:", bb.mean(), bb.var())
pmf_bb = bb.pmf(k)

plt.plot(k, pmf_nh, 'o', label='Negative hypergeometric')
plt.plot(k, pmf_bb, '.', label='Beta-binomial')
plt.title('PMF')
plt.xlabel('k')
plt.legend(framealpha=1, shadow=True)
plt.grid()
plt.show()
