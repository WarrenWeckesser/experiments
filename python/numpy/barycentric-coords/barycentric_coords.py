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
    """
    A = np.vstack((simplex.T, np.ones(simplex.shape[0])))
    return np.linalg.solve(A, pad1(p).T).T
