import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zipfian


m = 100_000_000
a = 0.85
n = 225

rng = np.random.default_rng()

x = zipfian.rvs(a, n, size=m, random_state=rng)

b = np.bincount(x, minlength=n + 1)[1:]
k = np.arange(1, n + 1)
pmf = zipfian.pmf(k, a, n)

plt.plot(k, pmf, 'o', alpha=0.25, label='Expected')
plt.plot(k, b/m, 'k.', alpha=0.75, label='Sample')
plt.grid(True)
plt.xlabel('k')
plt.legend(framealpha=1, shadow=True)
plt.title(f'Sample and Expected Probabilities\n{a=}, {n=}, {m=}')
plt.semilogy()
plt.show()
