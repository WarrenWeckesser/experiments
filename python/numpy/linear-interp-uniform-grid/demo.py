import numpy as np
import matplotlib.pyplot as plt
from linear_interp_uniform_grid import interp1du


rng = np.random.default_rng(0x1ce1cebab1e)
n = 25
y = rng.normal(size=n).cumsum()
x = np.arange(1, len(y) + 1)
x0 = x[0]
step = x[1] - x[0]

# Generate some random points at which to interpolate, possibly
# include points outside the interpolation interval.
nsamples = 80
xx = x0 + step*((len(y)+3)*rng.random(nsamples) - 2)

yy = interp1du(xx, x0, step, y)

plt.plot(x, y, 'o-', alpha=0.75, markersize=3.5, label='uniformly spaced data')
plt.plot(xx, yy, 'ro', alpha=0.75, markersize=2.5, label='interpolated values')
plt.legend(framealpha=1, shadow=True)
plt.grid(True)
plt.show()
# plt.savefig('demo.svg', transparent=True)
