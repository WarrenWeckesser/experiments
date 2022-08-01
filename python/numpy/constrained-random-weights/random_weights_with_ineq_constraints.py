
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


def _random_weights_rejection(bounds, nsamples, cmin, cmax, rng):
    """
    Generate random weights.

    A "weights" sample is a 1-d array of values in [0, 1] that sum to 1.

    For this function, the length of the array is set by ``len(bounds)``.
    ``bounds`` must be a 2-d NumPy array with shape ``(n, 2)``. Each
    row gives the lower and upper bound of the corresponding random weight.

    This is a rejection method.  ``nsamples`` is the number of *candidates*
    to be generated.  The number of samples returned will generally be
    less than ``nsamples``.
    """
    s = rng.uniform(cmin[:-1], cmax[:-1], size=(nsamples, len(bounds)-1))
    z = 1 - s.sum(axis=1)
    mask = (bounds[-1, 0] < z) & (z < bounds[-1, 1])

    samples = np.column_stack((s[mask], z[mask]))
    return samples


def random_weights(bounds, nsamples, rng=None):
    """
    Generate random weights.

    A "weights" sample is a 1-d array of values in [0, 1] that sum to 1.

    For this function, the length of the array is set by ``len(bounds)``.
    ``bounds`` must be a 2-d NumPy array with shape ``(n, 2)``. Each
    row gives the lower and upper bound of the corresponding random weight.

    This is a rejection method.  Be careful: the method implemented by this
    function has no lower bound on the acceptance rate; for some ``bounds``,
    the acceptance rate can be *very* low.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(889834848603455)
    >>> bounds = np.array([[0.1, 0.3], [0.2, 0.8], [0.05, 0.2], [0.05, 0.2]])
    >>> w = random_weights(bounds, 10, rng=rng)
    >>> w
    array([[0.2433808 , 0.48200472, 0.13659622, 0.13801826],
           [0.24732063, 0.53962185, 0.08603213, 0.12702538],
           [0.26126027, 0.45397118, 0.13229396, 0.15247459],
           [0.12556574, 0.64381902, 0.05610864, 0.17450659],
           [0.23442762, 0.48725061, 0.10074363, 0.17757814],
           [0.14484602, 0.61525839, 0.05651471, 0.18338088],
           [0.11788144, 0.69038204, 0.12077772, 0.07095881],
           [0.21745906, 0.53443103, 0.0702289 , 0.17788101],
           [0.23175572, 0.6135676 , 0.08310453, 0.07157215],
           [0.11296737, 0.65230007, 0.15007042, 0.08466214]])

    Check that the rows sum to 1:

    >>> w.sum(axis=1)
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

    """
    if rng is None:
        rng = np.random.default_rng()

    constrained_points = compute_intersection_vertices(bounds)
    cmin = constrained_points.min(axis=0)
    cmax = constrained_points.max(axis=0)

    samples = np.zeros((nsamples, len(bounds)))
    nfilled = 0
    while nfilled < nsamples:
        sample1 = _random_weights_rejection(bounds, nsamples - nfilled,
                                            cmin, cmax, rng)
        m = len(sample1)
        samples[nfilled:nfilled+m, :] = sample1
        nfilled += m
    return samples
