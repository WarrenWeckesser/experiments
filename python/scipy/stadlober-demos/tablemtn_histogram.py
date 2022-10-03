

import numpy as np
import matplotlib.pyplot as plt


def tablemtn(x, loc, scale):
    x = np.asarray(x)
    y = np.zeros(x.shape)
    z = (x - loc)/scale
    mask = (z >= -1) & (z <= 1)
    y[mask] = 0.25/scale
    y[~mask] = (0.25*scale / (x[~mask] - loc)**2)
    return y


u, v = np.random.uniform([[0], [-1]], 1, size=(2, 500000))
x = v/u

xbound = 5

nbins = 201
densities, edges, rectangles = plt.hist(x, bins=nbins,
                                        range=(-xbound, xbound), density=True,
                                        alpha=0.35)

# Adjust the plotted densities to account for the bounds applied
# to the histogram.
for rect in rectangles:
    rect.set_height((1 - 0.5/xbound)*rect.get_height())

xx = np.linspace(-xbound, xbound, 1000)
yy = tablemtn(xx, loc=0, scale=1)
plt.plot(xx, yy, 'k', linewidth=1)

plt.xlim(-xbound+1, xbound-1)

plt.grid(True)

plt.show()
