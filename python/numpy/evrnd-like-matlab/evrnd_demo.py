
import numpy as np
import matplotlib.pyplot as plt
from evrnd import evrnd


mu = 10
sigma = 2.5
n = 20000

x = evrnd(mu, sigma, n)

# Plot the normalized histogram of the sample.
plt.hist(x, bins=100, density=True, alpha=0.7)
plt.grid(alpha=0.25)
plt.xlabel('x')
plt.title(f'Histogram for mu={mu}, sigma={sigma}, n={n}')
# plt.show()
plt.savefig('evrnd_demo.png')
