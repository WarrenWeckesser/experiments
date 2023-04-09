
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from mixturen import MixtureN


# A mixture of two gamma distributions.
mx = MixtureN(gamma, gamma)

params = (1.1, 1, 8, 0, 1, 0, 1)

x = np.linspace(0, 20, 2501)
y = mx.pdf(x, *params)
sample = mx.rvs(*params, size=100000)

plt.plot(x, y)
plt.hist(sample, bins=80, density=True, alpha=0.3)
plt.xlim(x.min(), x.max())
plt.show()
