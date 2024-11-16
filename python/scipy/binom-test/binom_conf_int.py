
from scipy.stats import binom
from scipy.optimize import fsolve, brentq


def binom_conf_int(x, n, confidence_level=0.95, alternative='two-sided'):
    alternatives = ['two-sided', 'less', 'greater']
    if alternative not in alternatives:
        raise ValueError(f'alternative must be one of {alternatives!r}')
    if alternative == 'two-sided':
        alpha = (1 - confidence_level) / 2
        plow = brentq(lambda p: binom.sf(x-1, n, p) - alpha, 0, 1)
        phigh = brentq(lambda p: binom.cdf(x, n, p) - alpha, 0, 1)
    elif alternative == 'less':
        alpha = 1 - confidence_level
        plow = 0.0
        phigh = brentq(lambda p: binom.cdf(x, n, p) - alpha, 0, 1)
    elif alternative == 'greater':
        alpha = 1 - confidence_level
        plow = brentq(lambda p: binom.sf(x-1, n, p) - alpha, 0, 1)
        phigh = 1.0
    estimate = x/n
    return estimate, plow, phigh
