import numpy as np
from deconvolve_lstsq import deconvolve_lstsq
import matplotlib.pyplot as plt


rng = np.random.default_rng(121263137472525314065)

n = 500
x = np.zeros(n)
num_impulses = 48
indices = rng.integers(0, n, size=num_impulses)
x[indices] = rng.exponential(scale=10, size=num_impulses)

# h = np.array([-0.1, 0.5, 1.0, -1.25, 2.5, -1.25, 1.0, 0.5, -0.1])
# h = windows.bohman(27)[1:-1]
# h = windows.gaussian(27, 4)
t = np.linspace(0, 1, 25)
h = t*np.exp(-9*t)
h = h/h.sum()

noise_level = 0.1
y = np.convolve(x, h) + noise_level*rng.normal(size=len(x) + len(h) - 1)

m = len(y) - len(x) + 1
zpad = np.zeros(m-1)
xp = np.r_[x, zpad]

h_est = deconvolve_lstsq(x, y, symmetry=None)

plt.plot(h, 'k.', label='Original h')
plt.plot(h_est, 'o', alpha=0.5, label='Estimated h')
plt.grid()
plt.legend(framealpha=1, shadow=True)
plt.show()
