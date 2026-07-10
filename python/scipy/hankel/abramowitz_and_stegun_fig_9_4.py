# WIP - levels need some fine-tuning.

import numpy as np
from scipy.special import hankel1
import matplotlib.pyplot as plt


def phase(z):
    return np.atan2(z.imag, z.real)


## t0 = [-3.0975, 0.2545]
## phase_levels = np.arange(-10, 10)*t0/2
## r0 = 2.4505

x0 = -4.001
x1 = 3.001
y0 = -1.501
y1 = 1.501
gridsize = 2500
x, y = np.meshgrid(np.linspace(x0, x1, gridsize), np.linspace(y0, y1, gridsize))
z = x + 1j*y
w = hankel1(0, z)

fig = plt.figure(figsize=(8, 5))
plt.contour(np.abs(w), levels=27, linewidth=0.5, colors='k', origin='lower', extent=(x0, x1, y0, y1))
plt.contour(phase(w), levels=27, linewidth=0.5, colors='k', linestyle='--', origin='lower', extent=(x0, x1, y0, y1))

plt.plot([x0, 0], [0, 0], 'r-', alpha=0.5)
plt.grid(alpha=0.35)
plt.xlim(x0, x1)
plt.ylim(y0, y1)
plt.show()
