# Copyright 2022 Warren Weckesser

import numpy as np


def _fast_2d_crosstab(x, y, mx, my):
    """
    Fast crosstab for nonnegative integer arrays x and y.
    The function assumes x < mx and y < my (note the inequalities are strict).

    "fast" just means it is faster than the general-purpose function
    scipy.stats.contingency.crosstab.
    """
    z = my*x + y
    return np.bincount(z, minlength=mx*my).reshape((mx, my))


def _random_2d_table_from_marginals(marginal0, marginal1, size=None, rng=None):
    """
    Inputs are marginal sums associated with the two categorical
    variables.  E.g. if c is the table, then marginal0 is c.sum(axis=1)
    and marginal1 is c.sum(axis=0).

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
    x = np.repeat(np.arange(len0), marginal0)
    y = np.repeat(np.arange(len1), marginal1)

    tables = np.empty((nvars, len0, len1), dtype=int)
    for k in range(nvars):
        rng.shuffle(x)
        tables[k] = _fast_2d_crosstab(x, y, len0, len1)
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
    m0 = table.sum(axis=1)
    m1 = table.sum(axis=0)
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


def random_table_from_table(table, *, size=None, rng=None):
    """
    Generate a random contingency table.

    Given a contingency table, generate a new random table with the same
    distribution of variables as the given table.

    By default, one random table is returned.  If ``size`` is an integer,
    an array with shape ``(size,) + table.shape`` is returned, containing
    ``size`` random tables.

    The method used to generate the random table is simple but slow.
    Let ``m`` be the number of dimensions of ``table``, and let ``N``
    be the sum of the elements in the table.  Both the time and space
    complexities of the algorithm to generate one sample are O(``m*N``).

    There is a lot of literature on the efficient generation of random
    contingency tables that has been completely ignored here!
    """
    table = np.asarray(table)
    ndim = table.ndim
    axes = set(range(ndim))
    sums = []
    for k in range(ndim):
        a = tuple(axes - {k})
        # m is the sum over all axes except k.
        m = table.sum(axis=a)
        sums.append(m)
    return random_table(*sums, size=size, rng=rng)


def random_table(*sums, size=None, rng=None):
    """
    Generate a random contingency table.

    Given the sums of the levels of a set of categorical variables,
    generate a random contingency table that is consistent with those
    sums.

    * The number of categorical variables (and therefore the number
      of dimensions of the table) is ``len(sums)``.
    * The number of levels for categorical variable ``k`` is
      ``len(sums[k])``, so the shape of a random table will be
      ``tuple(len(s) for s in sums)``.
    * ``sums[k]`` gives the number of occurrences of each level for
      for categorical variable ``k``.  Another way to say this is that
      ``sums[k]`` is the marginal sum of the table over all axes except
      ``k``.

    By default, one random table is returned.  If ``size`` is an integer,
    an array with shape ``(size,) + table.shape`` is returned, containing
    ``size`` random tables.

    The method used to generate the random table is simple but slow. Let
    ``m`` be the number of variables (i.e. ``m = len(sums)``), and let
    ``N`` be the sum of the elements in the table (so ``N = sum(sums[0])``).
    Both the time and space complexities of the algorithm to generate one
    random table are O(``m*N``).

    There is a lot of literature on the efficient generation of random
    contingency tables that has been completely ignored here!

    Examples
    --------
    Among a group of 100 people, 87 are right-handed and 13 are
    left-handed.  Also, when asked about their favorite ice cream,
    34 said chocolate, 23 said vanilla, 13 said strawberry, 15 said
    some other flavor and 15 had no preference.

    There are two categorical variables: *handedness* and *flavor
    preference*.  Handedness has two levels: *right* and *left*.  Flavor
    preference has five levels: *chocolate*, *vanilla*, *strawberry*,
    *other* and *none*.  The sums for these variables are [87, 13] and
    [34, 23, 13, 15, 15], respectively.  To generate a random contingency
    table consistent with this data, we can do the following:

    >>> random_table([87, 13], [34, 23, 13, 15, 15])
    array([[28, 18, 13, 13, 15],
           [ 6,  5,  0,  2,  0]])

    To generate three such tables with one call of ``random_table``:

    >>> random_table([87, 13], [34, 23, 13, 15, 15], size=3)
    array([[[31, 21, 12, 10, 13],
            [ 3,  2,  1,  5,  2]],

           [[28, 20, 11, 14, 14],
            [ 6,  3,  2,  1,  1]],

           [[33, 20, 11, 10, 13],
            [ 1,  3,  2,  5,  2]]])

    """
    sums = [np.asarray(s) for s in sums]
    N = sums[0].sum()
    for s in sums[1:]:
        if s.sum() != N:
            raise ValueError('All the input sequences must have the same sum.')

    ndim = len(sums)
    if ndim == 2:
        return _random_2d_table_from_marginals(sums[0], sums[1],
                                               size=size, rng=rng)

    table_shape = tuple(len(s) for s in sums)

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

    data = np.empty((ndim, N), dtype=int)
    for k, s in enumerate(sums):
        data[k] = np.repeat(np.arange(len(s)), s)

    # Precompute stuff that will be used in the
    # function _fast_crosstab().
    maxes = data.max(axis=1) + 1
    cm = np.cumprod(maxes[::-1])
    maxprod = cm[-1]
    coeffs = np.r_[[1], cm[:-1]][::-1]

    tables = np.empty((nvars,) + table_shape, dtype=int)
    for k in range(nvars):
        rowshuffle(data[:-1])
        tables[k] = _fast_crosstab(data, maxes, maxprod, coeffs)

    if size is None:
        tables = tables[0]

    return tables
