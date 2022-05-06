
import numpy as np
import matplotlib.pyplot as plt
from random_weights_with_ineq_constraints import random_weights, check_bounds


bounds = np.array([[0.045, 0.150],
                   [0.045, 0.160],
                   [0.060, 0.200],
                   [0.070, 0.200],
                   [0.080, 0.120],
                   [0.080, 0.150],
                   [0.080, 0.200],
                   [0.000, 0.030],
                   [0.000, 0.035],
                   [0.090, 0.175]])

check_bounds(bounds)
nrequested = 1000000
samples = random_weights(bounds, 1000000)

nactual = len(samples)
print(f"len(samples): {len(samples)}")
print(f"Acceptance rate: {nactual/nrequested}")

# Check
assert np.allclose(samples.sum(axis=1), 1, rtol=1e-12, atol=0)
assert np.all((bounds[:, 0] <= samples) & (samples <= bounds[:, 1]))


for k in range(samples.shape[1]):
    plt.hist(samples[:, k], bins=40, alpha=0.35, density=True)

plt.title('Marginal Histograms\n(normalized with density=True)')
plt.grid(alpha=0.2)
plt.show()
