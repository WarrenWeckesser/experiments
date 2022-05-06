
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
    Generate random weights.

    A "weights" sample is a 1-d array of values in [0, 1] that sum to 1.

    For this function, the length of the array is set by ``len(bounds)``.
    ``bounds`` must be a 2-d NumPy array with shape ``(n, 2)``. Each
    row gives the lower and upper bound of the corresponding random weight.

    This is a rejection method.  ``nsamples`` is the number of *candidates*
    to be generated.  The number of samples returned will generally be
    less than ``nsamples``.  Be careful: the method implemented by this
    function has no lower bound on the acceptance rate; for some ``bounds``,
    the acceptance rate can be *very* low.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(889834848603455)
    >>> bounds = np.array([[0.1, 0.2], [0.2, 0.8], [0.05, 0.2], [0.05, 0.1]])
    >>> w = random_weights(bounds, 10, rng=rng)
    >>> w
    array([[0.1716904 , 0.60920283, 0.13659622, 0.08251055],
           [0.17366032, 0.64377311, 0.08603213, 0.09653444],
           [0.18063014, 0.59238271, 0.13229396, 0.0946932 ]])

    Note that 10 samples were requested but only 3 were generated.

    Check that the rows sum to 1:

    >>> w.sum(axis=1)
    array([1., 1., 1.])

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
