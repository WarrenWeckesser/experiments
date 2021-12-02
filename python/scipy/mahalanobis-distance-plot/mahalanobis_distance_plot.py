# https://stackoverflow.com/questions/44998025/
#     selecting-points-in-dataset-that-belong-to-a-multivariate-gaussian-distribution

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


# Mean
M = [10, 7]

# Covariance matrix
V = np.array([[ 9, -2],
              [-2,  2]])
VI = np.linalg.inv(V)

# Number of samples
n = 1000

# Generate a sample from the multivariate normal distribution
# with mean M and covariance matrix V.
rng = np.random.default_rng()
x = rng.multivariate_normal(M, V, size=n)

# Compute the Mahalanobis distance of each point in the sample.
mdist = cdist(x, [M], metric='mahalanobis', VI=VI)[:, 0]

# Find where the Mahalanobis distance is less than R.
R = 1.5
d2_mask = mdist < R
x2 = x[d2_mask]

plt.plot(x2[:, 0], x2[:, 1], 'o',
         markeredgecolor='r', markerfacecolor='w', markersize=6, alpha=0.25)
plt.plot(x[:, 0], x[:, 1], 'k.', markersize=5, alpha=0.3)
plt.grid(alpha=0.3)
plt.axis('equal')
tx = np.linspace(x[:, 0].min(), x[:, 0].max(), 250)
ty = np.linspace(x[:, 1].min(), x[:, 1].max(), 250)
X, Y = np.meshgrid(tx, ty)
points = np.column_stack((X.ravel(), Y.ravel()))
Z = cdist(points, [M], metric='mahalanobis', VI=VI)
plt.contour(tx, ty, Z.reshape(250, 250), levels=[R], alpha=0.25, colors=['r'])
plt.title(f'Circled points: Mahalanobis distance to (10, 7) is less than {R}')

plt.savefig('images/mahalanobis-distance-plot.png')

# plt.show()
