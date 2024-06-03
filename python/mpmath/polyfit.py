#
# An incomplete translation of numpy's polynomial fitting code.
#

from mpmath import mp


def _matrix_col_to_list(m, col):
    """
    Convert column `col` of mp.matrix `m` to a 1-d list.
    """
    return m[:, col].T.tolist()[0]


def scale_rows(m, scale):
    p = mp.matrix(m.rows, m.cols)
    for i in range(m.rows):
        s = scale[i]
        for j in range(m.cols):
            p[i, j] = m[i, j]/s
    return p


def scale_cols(m, scale):
    p = mp.matrix(m.rows, m.cols)
    for j in range(m.cols):
        s = scale[j]
        for i in range(m.rows):
            p[i, j] = m[i, j]/s
    return p


def msum(m, axis):
    if axis == 0:
        sums = []
        for j in range(m.cols):
            s = mp.zero
            for i in range(m.rows):
                s += m[i, j]
            sums.append(s)
        return sums
    else:
        # axis == 1
        sums = []
        for i in range(m.rows):
            s = mp.zero
            for j in range(m.cols):
                s += m[i, j]
            sums.append(s)
        return sums


def mshape(m):
    return m.rows, m.cols


def vander(x, deg):
    n = len(x)
    m = mp.matrix(n, deg + 1)
    for i, x1 in enumerate(x):
        for j in range(deg + 1):
            m[i, j] = x1**j
    return m


def _lstsq(A, b):
    U, S, V = mp.svd(A, compute_uv=True)
    s = _matrix_col_to_list(S, 0)
    threshold = s[0]*max(A.rows, A.cols)*mp.eps
    s1 = [s[k] for k in range(len(s)) if abs(s[k]) > threshold]
    beta = _matrix_col_to_list(U.transpose() @ mp.matrix(b), 0)
    t1 = [beta[k]/s1[k] for k in range(len(s1))]
    z = mp.matrix(t1 + [mp.zero]*(V.rows - len(t1)))
    return V.transpose() @ z


@mp.extradps(5)
def polyfit(x, y, deg, rcond=None, full=False, w=None):
    """
    Fit a polynomial to the given data.
    """
    if rcond is not None:
        raise NotImplementedError('rcond is not implemented')

    x = [mp.mpf(t) for t in x]
    y = [mp.mpf(t) for t in y]

    if deg < 0:
        raise ValueError("deg must be nonnegative")

    n = len(x)
    if n != len(y):
        raise TypeError("x and y must have the same length")

    lmax = deg
    # order = deg + 1
    van = vander(x, lmax)

    if w is not None:
        raise NotImplementedError('weights not implemented')
    # if w is not None:
    #     w = np.asarray(w) + 0.0
    #     if w.ndim != 1:
    #         raise TypeError("expected 1D vector for w")
    #     if len(x) != len(w):
    #         raise TypeError("expected x and w to have same length")
    #     # apply weights. Don't use inplace operations as they
    #     # can cause problems with NA.
    #     lhs = lhs * w
    #     rhs = rhs * w

    # if rcond is None:
    #     rcond = n*mp.eps

    # Determine the norms of the design matrix columns.
    # if issubclass(lhs.dtype.type, np.complexfloating):
    #     scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
    # else:
    #     scl = np.sqrt(np.square(lhs).sum(1))
    # XXX For now, only real inputs are handled.
    m2 = van.apply(lambda t: t**2)
    sums = msum(m2, axis=0)
    scale = [mp.sqrt(t) if t > 0 else mp.zero for t in sums]

    # # Solve the least squares problem.
    # c, resids, rank, s = np.linalg.lstsq(lhs.T/scl, rhs.T, rcond)
    # c = (c.T/scl).T

    A = scale_cols(van, scale)
    v = _lstsq(A, y)
    c = [v1/scale1 for v1, scale1 in zip(v, scale)]
    return c

    # # Expand c to include non-fitted coefficients which are set to zero
    # if deg.ndim > 0:
    #     if c.ndim == 2:
    #         cc = np.zeros((lmax+1, c.shape[1]), dtype=c.dtype)
    #     else:
    #         cc = np.zeros(lmax+1, dtype=c.dtype)
    #     cc[deg] = c
    #     c = cc

    # # warn on rank reduction
    # if rank != order and not full:
    #     msg = "The fit may be poorly conditioned"
    #     warnings.warn(msg, RankWarning, stacklevel=2)

    # if full:
    #     return c, [resids, rank, s, rcond]
    # else:
    #     return c
