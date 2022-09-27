
from scipy.special import betainc
from scipy.optimize import fsolve


def _beta_select_equation(params, p1, x1, p2, x2):
    return betainc(*params, [x1, x2]) - [p1, p2]


def beta_select(p1, x1, p2, x2):
    """
    Compute the standard beta distibution parameters from a pair of quantiles.

    This function is roughly equivalent to the R function `beta.select()` from
    the LearnBayes R package.

    The function finds the parameters alpha and beta of the beta distribution
    such that::

        CDF(x1; alpha, beta) = p1
        CDF(x2; alpha, beta) = p2.

    Examples
    --------
    >>> alpha, beta = beta_select(p1=0.5, x1=0.25, p2=0.9, x2=0.45)
    >>> alpha
    2.6689738643869267
    >>> beta
    7.364790585308813
    """
    params, info, status, mesg = fsolve(_beta_select_equation, [1, 1],
                                        args=(p1, x1, p2, x2), xtol=1e-12,
                                        full_output=True)
    if status != 1:
        raise RuntimeError(f'fsolve failed: {mesg}')
    return params
