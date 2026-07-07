import math
from scipy.special import nctdtr, stdtrit, ndtri
from scipy.optimize import fsolve


def _n_eq_twosided(n, alpha_term, beta, ncfactor):
    # alpha_term = 1 - alpha/2 
    # beta = 1 - power
    # ncfactor = abs(delta) / sqrt(2) / sigma
    n = n[0]
    df = 2 * (n - 1)
    nc = ncfactor * math.sqrt(n)
    t = stdtrit(df, alpha_term)
    return nctdtr(df, nc, -t) - nctdtr(df, nc, t) + beta


def _n_eq_onesided(n, alpha_term, beta, ncfactor):
    # alpha_term = 1 - alpha
    # beta = 1 - power
    # ncfactor = abs(delta) / sqrt(2) / sigma
    n = n[0]
    df = 2 * (n - 1)
    nc = ncfactor * math.sqrt(n)
    t = stdtrit(df, alpha_term)
    return beta - nctdtr(df, nc, t)


def find_n(*, alpha, power, sigma, delta, alternative='two-sided'):
    """
    t-test power calculation.

    This function does the same computation as power.t.test() in R for the
    case where n is computed.

    Returns n, the number in each group.
    """
    #
    # XXX/TODO: Check validity of the one-sided calculation when delta < 0.
    #

    if alternative not in ['one-sided', 'two-sided']:
        raise ValueError("alternative must be 'one-sided' or 'two-sided'")

    if alternative == 'two-sided':
        n_eq = _n_eq_twosided
        alpha_term = 1 - alpha / 2
    else:
        n_eq = _n_eq_onesided
        alpha_term = 1 - alpha

    # Normal approximation to estimate n.
    n0 = 2 * ((ndtri(alpha_term) + ndtri(power)) * sigma / delta) ** 2

    beta = 1 - power
    ncfactor = abs(delta) / math.sqrt(2) / sigma
    result = fsolve(n_eq, x0=n0, args=(alpha_term, beta, ncfactor))
    return result.item(0)
