
from math import sqrt
from scipy.special import ndtri


def _wilson_method(k, n, *, confidence_level=0.95, alternative='two-sided',
                   correction=False):
    # This function assumes that the arguments have already been validated.
    p = k / n
    if alternative == 'two-sided':
        z = ndtri(0.5 + 0.5*confidence_level)
    else:
        z = ndtri(confidence_level)

    t = 1 + z**2/n
    r = (p + z**2/(2*n)) / t

    if correction:
        if alternative == 'less' or k == 0:
            lo = 0.0
        else:
            dlo = (z * sqrt(z**2 - 1/n + 4*n*p*(1 - p) + (4*p - 2)) + 1) / (2*n*t)
            lo = r - dlo
        if alternative == 'greater' or k == n:
            hi = 1.0
        else:
            dhi = (z * sqrt(z**2 - 1/n + 4*n*p*(1 - p) - (4*p - 2)) + 1) / (2*n*t)
            hi = r + dhi
    else:
        d = z/t * sqrt(p*(1-p)/n + (z/(2*n))**2)
        if alternative == 'less' or k == 0:
            lo = 0.0
        else:
            lo = r - d
        if alternative == 'greater' or k == n:
            hi = 1.0
        else:
            hi = r + d

    return p, lo, hi
