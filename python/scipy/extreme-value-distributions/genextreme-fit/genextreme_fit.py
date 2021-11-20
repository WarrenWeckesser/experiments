
import numpy as np
from scipy.stats import genextreme
import matplotlib.pyplot as plt


seed = 485394758879878435
rng = np.random.default_rng(seed)

n = 5000  # Number of block maxima to compute.
m = 20    # Block size

a = 3.8      # gamma distribution shape parameter
scale = 2.0  # gamma distribution scale parameter

samples = rng.gamma(a, scale=scale, size=(n, m)).max(axis=-1)

params = genextreme.fit(samples)
print(params)

plt.hist(samples, bins=60, density=True, color='c', alpha=0.75,
         label='normalized histogram')
xx = np.linspace(0, samples.max(), 400)
yy = genextreme.pdf(xx, *params)
plt.plot(xx, yy, 'k', linewidth=1.5, alpha=0.6,
         label="genextreme.fit(samples)")
plt.legend()
plt.grid(alpha=0.2)
plt.title('Fit genextreme to block maxima of the gamma distribution\n'
          f'gamma parameters: a={a}, scale={scale}\n'
          f'block size: {m}')
plt.tight_layout()
# plt.show()
plt.savefig('genextreme_fit.png', dpi=125)
