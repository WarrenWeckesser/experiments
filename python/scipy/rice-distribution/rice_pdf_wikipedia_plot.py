import numpy as np
from scipy.stats import rice
import matplotlib.pyplot as plt


# Generate the plot of the Rice PDF shown in the wikipedia article.

sigma = 1.0
nus = [0, 0.5, 1.0, 2.0, 4.0]
clrs = ['b', 'g', 'k', 'r', 'm']

x = np.linspace(0, 8, 250)
for nu, clr in zip(nus, clrs):
    p = rice.pdf(x, nu, loc=0, scale=sigma)
    plt.plot(x, p, clr, linewidth=2, label=rf"$\nu$ = {nu:3.1f}")

plt.grid()
plt.title(rf"Rice PDF, $\sigma$ = {sigma:4.2f}")
plt.legend()
plt.show()
