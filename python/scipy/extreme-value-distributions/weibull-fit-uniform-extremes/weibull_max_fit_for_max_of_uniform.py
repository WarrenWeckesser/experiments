
import numpy as np
from scipy.stats import weibull_max
import matplotlib.pyplot as plt


seed = 48394759879878435
rng = np.random.default_rng(seed)

# Generate N samples, each of which is the maximum of m samples
# from uniform distribution
N = 4000
m = 10
samples = rng.uniform(size=(N, m)).max(axis=1)

c, loc, scale = weibull_max.fit(samples, floc=1)
print(f'weibull_max fit: c={c}, scale={scale}')

xx = np.linspace(0.95*samples.min(), 1, 1000)
pdf = weibull_max.pdf(xx, c, loc=loc, scale=scale)

# Plots
plt.hist(samples, bins=40, density=True, alpha=0.6,
         label="normalized histogram of samples")

plt.plot(xx, pdf, 'k', alpha=0.75,
         label="PDF of weibull_max fit")

plt.text(1 - 0.5*samples.max(), 0.7*pdf.max(),
         f'weibull_max fit: c={c:.5f}, scale={scale:.5f}',
         bbox=dict(facecolor='gray', alpha=0.4))

plt.title('Fit weibull_max to block maxima of samples from Uniform(0, 1)\n'
          f'(block size: {m})')
plt.legend(framealpha=1, shadow=True)
# plt.show()
plt.savefig('weibull_max_fit_for_max_of_uniform.png', dpi=100)
