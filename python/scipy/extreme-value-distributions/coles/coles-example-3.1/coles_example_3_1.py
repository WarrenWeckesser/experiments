# See Coles Example 3.1.  Here the underlying distribution is the exponential
# distribution with a given scale; scale=1 is used in Example 3.1.

import numpy as np
from scipy.stats import expon, gumbel_r
import matplotlib.pyplot as plt


seed = 782347878783482347
rng = np.random.default_rng(seed)

scale = 1
block_size = 2500
num_blocks = 10000
smax = expon.rvs(scale=scale, size=(num_blocks, block_size),
                 random_state=rng).max(axis=1)

max_result = gumbel_r.fit(smax)

expected_loc = scale*np.log(block_size)
print("Fit parameters: %8.4f  %8.4f" % max_result)
print("Expected:       %8.4f  %8.4f" % (expected_loc, scale))


plt.hist(smax, density=True, bins=50, color='g', ec='none', alpha=0.5,
         label='histogram')

x = np.linspace(0, smax.max(), 200)
p = gumbel_r.pdf(x, *max_result)
plt.plot(x, p, 'k', alpha=0.6, lw=1.5, label='gumbel_r fit')
plt.title('Fit gumbel_r to block maxima of exponential variates\n'
          f'(scale: {scale}, block size: {block_size})')
plt.legend()
text = ('               loc     scale\n' +
        f'Fit:      {max_result[0]:8.4f}  {max_result[1]:8.4f}\n' +
        f'Expected: {expected_loc:8.4f}  {scale:8.4f}')
plt.text(0.82*max_result[0] + 0.18*smax.max(), 0.7*p.max(), text,
         fontfamily='monospace', bbox=dict(facecolor='gray', alpha=0.25))
# plt.show()
plt.savefig('coles_example_3_1.png', dpi=125)
