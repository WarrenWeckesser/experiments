
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt


seed = 485394759879878435
rng = np.random.default_rng(seed)

# Generate N samples, each of which is the minimum of m samples
# from uniform distribution
N = 4000
m = 10
samples = rng.uniform(size=(N, m)).min(axis=1)

c, loc, scale = weibull_min.fit(samples, floc=0)
print(f'weibull_min fit: c={c}, scale={scale}')

xx = np.linspace(0, 1.05*samples.max(), 1000)
pdf = weibull_min.pdf(xx, c, loc=loc, scale=scale)

# Plots
plt.hist(samples, bins=40, density=True, alpha=0.6,
         label="normalized histogram of samples")
plt.plot(xx, pdf, 'k', alpha=0.75,
         label="PDF of weibull_min fit")

plt.text(0.25*samples.max(), 0.7*pdf.max(),
         f'weibull_min fit: c={c:.5f}, scale={scale:.5f}',
         bbox=dict(facecolor='gray', alpha=0.4))

plt.title('Fit weibull_min to block minima of samples from Uniform(0, 1)\n'
          f'(block size: {m})')
plt.legend(framealpha=1, shadow=True)
# plt.show()
plt.savefig('weibull_min_fit_for_min_of_uniform.png', dpi=100)
