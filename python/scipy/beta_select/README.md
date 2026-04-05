The function `beta_select(p1, x1, p2, x2)` defined in `beta_select.py`
is roughly equivalent to the R function `beta.select()` from the
LearnBayes R package.

The function finds the parameters `a` and `b` of the beta distribution
such that::

    CDF(x1; a, b) = p1
    CDF(x2; a, b) = p2

For example, the following finds the parameters of the beta distribution for which
the CDF(0.25; a, b) = 0.5 and CDF(0.45; a, b) = 0.9.

    >>> x1 = 0.25
    >>> x2 = 0.45
    >>> a, b = beta_select(p1=0.5, x1=x1, p2=0.9, x2=x2)
    >>> a
    np.float64(2.6689738643869356)
    >>> b
    np.float64(7.364790585308838)

    >>> from scipy.stats import beta
    >>> beta.cdf(x1, a, b)
    np.float64(0.5)
    >>> beta.cdf(x2, a, b)
    np.float64(0.9000000000000001)
