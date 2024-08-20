
import numpy as np


def median2(x, axis=-1):
    """
    Compute the median of x along the given axis, returning two values.

    If the length of the input is odd, the two values are the same, and
    they are the standard median.

    If the length is even, the two values might be different. They are
    the values that would be at positions n//2 - 1 and n//2 of the sorted
    input, where n is the length of the input. For numerical data types,
    the standard median is the average of the two values.

    The returned array has the same data type as the input.

    Examples
    --------
    >>> import numpy as np

    >>> x = np.array([[8, 0, 6, 0, 3, 8, 7, 4, 0, 4],
    ...               [5, 6, 3, 6, 8, 1, 3, 0, 7, 1],
    ...               [0, 2, 7, 9, 9, 0, 1, 9, 8, 1]])
    >>> median2(x)
    array([[4, 4],
           [3, 5],
           [2, 7]])
    >>> median2(x, axis=0)
    array([[5, 2, 6, 6, 8, 1, 3, 4, 7, 1],
           [5, 2, 6, 6, 8, 1, 3, 4, 7, 1]])


    >>> d = np.array(['1986-03-16T20:01:27', '2002-04-27T09:05:24',
    ...               '2015-07-23T20:18:53', '2017-03-04T20:02:12',
    ...               '1988-09-27T18:14:47', '2005-03-23T13:09:49'],
    ...              dtype='datetime64[s]')
    >>> median2(d)
    array(['2002-04-27T09:05:24', '2005-03-23T13:09:49'],
          dtype='datetime64[s]')

    >>> s = np.array(["ab", "d", "d", "e", "a", "b", "b", "def"],
    ...              dtype=np.dtypes.StringDType())
    >>> median2(s)
    array(['b', 'd'], dtype=StringDType())

    """
    x = np.asarray(x)
    a = np.swapaxes(x, axis, -1)
    shp = a.shape
    out = np.empty(a.shape[:-1] + (2,), dtype=a.dtype)
    n = shp[-1]
    m = (n - 1) // 2
    ap = np.partition(a, m)
    out[..., 0] = ap[..., m]
    if n & 1:
        out[..., 1] = ap[..., m]
    else:
        out[..., 1] = ap[..., m+1:].min(axis=-1)
    return np.swapaxes(out, -1, axis)
