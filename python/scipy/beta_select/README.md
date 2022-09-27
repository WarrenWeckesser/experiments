The function `beta_select(p1, x1, p2, x2)` defined in `beta_select.py`
is roughly equivalent to the R function `beta.select()` from the
LearnBayes R package.

The function finds the parameters alpha and beta of the beta distribution
such that::

    CDF(x1; alpha, beta) = p1
    CDF(x2; alpha, beta) = p2.

For example::

    >>> alpha, beta = beta_select(p1=0.5, x1=0.25, p2=0.9, x2=0.45)
    >>> alpha
    2.6689738643869267
    >>> beta
    7.364790585308813
