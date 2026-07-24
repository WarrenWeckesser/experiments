
def chebyshev_lobatto_sample(mp, func, a, b, N):
    t = [mp.cos(mp.pi * k / N) for k in range(N + 1)]
    mid = (a + b)/2
    h = (b - a)/2
    x = [mid + h*t1 for t1 in t]
    y = [func(x1) for x1 in x]
    return t, x, y


def chebyshev_coefficients(mp, y, N):
    """
    Direct DCT-I applied to samples at Chebyshev-Lobatto points.

    Parameters
    ----------
    mp : mpmath.ctx_mp.MPContext
        mpmath context.  This can be `mpmath.mp` in most cases.
    y : sequence of values
        Values of the function being approximated.
    N : int
        Order of the Chebyshev polynomial
    """ 
    # XXX Verify that the coefficient normalization is correct.
    c = []
    for j in range(N + 1):
        s = mp.mpf('0')
        for k in range(N + 1):
            w = mp.mpf('0.5') if (k == 0 or k == N) else mp.mpf('1')
            s += w * y[k] * mp.cos(mp.pi * j * k / N)
        c.append(2 * s / N)

    # Conventional Chebyshev scaling
    c[0] /= 2
    c[-1] /= 2
    return c


def chebyshev_eval(x, c):
    """
    Evaluate a Chebyshev series using Clenshaw's algorithm.

    Computes:
        p(x) = c[0] + c[1]*T_1(x) + ... + c[n]*T_n(x)

    Parameters
    ----------
    mp : mpmath.ctx_mp.MPContext
        mpmath context.  This can be `mpmath.mp` in most cases.
    x : float
        Point in [-1, 1].
    c : sequence of float
        Chebyshev coefficients.

    Returns
    -------
    float
        Polynomial value.
    """
    n = len(c) - 1

    b_kplus1 = 0.0
    b_kplus2 = 0.0

    for k in range(n, 0, -1):
        b_k = 2.0*x*b_kplus1 - b_kplus2 + c[k]
        b_kplus2 = b_kplus1
        b_kplus1 = b_k

    return x*b_kplus1 - b_kplus2 + c[0]
