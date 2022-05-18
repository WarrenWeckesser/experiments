# Copyright 2022 Warren Weckesser

import numpy as np


def _fast_2d_crosstab(x, y, mx, my):
    """
    Fast crosstab for nonnegative integer arrays x and y.
    The function assumes x < mx and y < my (note the inequalities are strict).
    """
    z = my*x + y
    return np.bincount(z, minlength=mx*my).reshape((mx, my))


def _random_2d_table_from_marginals(marginal0, marginal1, size=None, rng=None):
    """
    Inputs are marginal sums associated with axis 0 and 1, resp.
    E.g. if c is the table, then marginal0 is c.sum(axis=0)
    and marginal1 is c.sum(axis=1).

    This is a slow way to generate random 2-d contingency tables.
    """
    if rng is None:
        rng = np.random.default_rng()

    if size is None:
        nvars = 1
    else:
        nvars = size

    n = np.sum(marginal0)
    if np.sum(marginal1) != n:
        raise ValueError('sum(marginal0) must equal sum(marginal1)')

    len0 = len(marginal0)
    len1 = len(marginal1)
    x = np.repeat(np.arange(len1), marginal1)
    y = np.repeat(np.arange(len0), marginal0)

    tables = np.empty((nvars, len1, len0), dtype=int)
    for k in range(nvars):
        rng.shuffle(x)
        tables[k] = _fast_2d_crosstab(x, y, len1, len0)
    if size is None:
        tables = tables[0]
    return np.array(tables)


def _random_2d_table(table, size=None, rng=None):
    """
    Generate a random contingency table.

    Given a 2-d contingency table, generate a random table with
    the same marginal sums.

    `table` must be a numpy array.
    """
    if table.ndim != 2:
        raise ValueError('table must be a 2-d array')
    m0 = table.sum(axis=0)
    m1 = table.sum(axis=1)
    return _random_2d_table_from_marginals(m0, m1, size=size, rng=rng)


def _fast_crosstab(data, maxes, maxprod, coeffs):
    # "fast" just means it is faster than the general-purpose function
    # scipy.stats.contingency.crosstab.
    # maxes, maxprod and coeffs are implicit in data, but for efficiency
    # it is assumed they have been computed before calling this function.
    #   maxes = data.max(axis=1) + 1
    #   cm = np.cumprod(maxes[::-1])
    #   maxprod = cm[-1]
    #   coeffs = np.r_[[1], cm[:-1]][::-1]
    z = coeffs @ data
    return np.bincount(z, minlength=maxprod).reshape(maxes)


def random_table(table, size=None, rng=None):
    """
    Generate a random contingency table.

    Given a contingency table, generate a new random table with the same
    distribution of variables as the given table.

    By default, one random table is returned.  If ``size`` is an integer,
    an array with shape ``(size,) + table.shape`` is returned, containing
    ``size`` random tables.

    The method used to generate the random table is simple but *slow*.
    Let ``m`` be the number of dimensions of ``table``, and let ``N``
    be the sum of the elements in the table.  Both the time and space
    complexities of the algorithm to generate one sample are O(``m*N``).

    There is a lot of literature on the efficient generation of random
    contingency tables that has been completely ignored here!
    """
    table = np.asarray(table)
    ndim = table.ndim
    if ndim == 2:
        # The 2-d specialization is a bit faster than the
        # generic n-d code in this function.
        return _random_2d_table(table, size=size, rng=rng)

    if rng is None:
        rng = np.random.default_rng()
    if hasattr(rng, 'permuted'):

        def rowshuffle(data):
            rng.permuted(data, axis=1, out=data)

    else:

        def rowshuffle(data):
            for row in data:
                rng.shuffle(row)

    if size is None:
        nvars = 1
    else:
        nvars = size

    axes = set(range(ndim))
    sums = []
    for k in range(ndim):
        a = tuple(axes - {k})
        # m is the sum over all axes except k.
        m = table.sum(axis=a)
        sums.append(m)

    data = np.empty((ndim, table.sum()), dtype=int)
    for k, s in enumerate(sums):
        data[k] = np.repeat(np.arange(len(s)), s)

    # Precompute stuff that will be used in the
    # function _fast_crosstab().
    maxes = data.max(axis=1) + 1
    cm = np.cumprod(maxes[::-1])
    maxprod = cm[-1]
    coeffs = np.r_[[1], cm[:-1]][::-1]

    tables = np.empty((nvars,) + table.shape, dtype=int)
    for k in range(nvars):
        rowshuffle(data[:-1])
        tables[k] = _fast_crosstab(data, maxes, maxprod, coeffs)
    if size is None:
        tables = tables[0]
    return tables
