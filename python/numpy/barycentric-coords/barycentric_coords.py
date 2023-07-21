import numpy as np


def pad1(x):
    padshape = x.shape[:-1] + (1,)
    return np.hstack((x, np.ones(padshape)))


def barycentric_coords(p, simplex):
    """
    Given a point (or points) in n-dimensional space, and a simplex
    in n-dimensional space (represented as n+1 points), compute the
    barycentric coordinates of the points with respect to the simplex.

    Parameters
    ----------
    p: array with shape (n,) or (m, n)
        Find the coordinates for these points.
    simplex: array with shape (n+1, n)
        The simplex, represented as n+1 points.

    Returns
    -------
    Array with shape (n,) or (m, n) holding the barycentric coordinates
    of each input point.

    Examples
    --------
    >>> import numpy as np
    >>> from barycentric_coords import barycentric_coords

    >>> simplex = np.array([[1.0, 3.0], [5.0, 2.0], [2.0, -3]])
    >>> pts = np.array([[2.0, 2.0], [1.0, 3.0], [4.0, -1.0], [2.5, 0]])
    >>> w = barycentric_coords(pts, simplex)
    >>> print(w)
    [[ 0.65217391  0.2173913   0.13043478]
     [ 1.          0.          0.        ]
     [-0.17391304  0.60869565  0.56521739]
     [ 0.2826087   0.26086957  0.45652174]]

    """
    return np.linalg.solve(pad1(simplex).T, pad1(p).T).T
