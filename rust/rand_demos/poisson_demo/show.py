
import numpy as np
import matplotlib.pyplot as plt


x = np.loadtxt('out', dtype=np.int64)
xmin = np.min(x)
xptp = np.ptp(x)
print(f'{xptp = }')
b = np.bincount(x - xmin)
print(f'{len(b) = }')

plt.plot(xmin + np.arange(xptp + 1), b, 'o', alpha=0.4)
plt.grid()
plt.show()
