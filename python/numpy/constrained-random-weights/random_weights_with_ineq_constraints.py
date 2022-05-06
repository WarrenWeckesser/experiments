
from itertools import combinations
import numpy as np


def compute_hypercube_edge_coordinate_pairs(n):
    # XXX This will be very slow for large n.
    p = 2**np.arange(n)
    pairs = []
    for m in range(n+1):
        for start in combinations(p, m):
            for c in combinations(sorted(list(set(p) - set(start))), 1):
                pairs.append((sum(start),
                              sum(sorted(list(set(c) | set(start))))))
    return pairs


def neighbors(k):
    for j in range(len(k)):
        if k[j] == 0:
            yield k[:j] + (1,) + k[j+1:]


def check_bounds(bounds):
    sum0 = bounds.sum(axis=0)
    if sum0[0] > 1:
        raise ValueError('weights within the given bounds not possible: '
                         f'sum of lower bounds ({sum0[0]}) exceeds 1')
    if sum0[1] < 1:
        raise ValueError('weights within the given bounds not possible: '
                         f'sum of upper bounds ({sum0[1]}) is less than 1')
    if np.isclose(sum0[0], 1, rtol=1e-12, atol=0):
        raise ValueError('random weights not possible; weights must all equal '
                         f'lower bounds {bounds[:, 0]}')
    if np.isclose(sum0[1], 1, rtol=1e-12, atol=0):
        raise ValueError('random weights not possible; weights must all equal '
                         f'upper bounds {bounds[:, 1]}')


def constraint_box_vertices(bounds):
    n = len(bounds)
    icoords = ((np.arange(2**n)[:, None] &
               (1 << np.arange(n))) > 0).astype(int)
    vertices = bounds[np.arange(n), icoords]
    return icoords, vertices


def compute_intersection_vertices(bounds):
    c, vertices = constraint_box_vertices(bounds)
    n = len(bounds)
    constrained_points = []
    pairs = compute_hypercube_edge_coordinate_pairs(n)
    for k0, k1 in pairs:
        v0 = vertices[k0]
        v1 = vertices[k1]
        s0 = sum(v0)
        s1 = sum(v1)
        if s0 <= 1 and s1 > 1:
            rho = (1 - s0)/(s1 - s0)
            constrained_points.append(v0 + rho*(v1 - v0))
    return np.array(constrained_points)


def random_weights(bounds, nsamples, rng=None):
    """
    This is a rejection method.  nsamples is the number of candidates
    to be generated.  The number of samples returned will generally be
    less than nsamples.
    """
    if rng is None:
        rng = np.random.default_rng()

    constrained_points = compute_intersection_vertices(bounds)
    cmin = constrained_points.min(axis=0)
    cmax = constrained_points.max(axis=0)

    s = rng.uniform(cmin[:-1], cmax[:-1], size=(nsamples, bounds.shape[0]-1))
    z = 1 - s.sum(axis=1)
    mask = (bounds[-1, 0] < z) & (z < bounds[-1, 1])

    samples = np.column_stack((s[mask], z[mask]))
    return samples


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


bounds = np.array([[0.04, 0.15],
                   [0.04, 0.16],
                   [0.04, 0.21],
                   [0.07, 0.20],
                   [0.07, 0.21],
                   [0.08, 0.12],
                   [0.08, 0.15],
                   [0.08, 0.20],
                   [0.00, 0.02],
                   [0.09, 0.18]])


"""
bounds = np.array([[0.1, 0.40],
                   [0.2, 0.50],
                   [0.3, 0.40]])
"""

check_bounds(bounds)


samples = random_weights(bounds, 1000000)

# Check
assert np.allclose(samples.sum(axis=1), 1, rtol=1e-12, atol=0)
assert np.all((bounds[:, 0] <= samples) & (samples <= bounds[:, 1]))
