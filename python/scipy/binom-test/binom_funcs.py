

from scipy.special import comb


def binom_pmf(n, p):
    return [comb(n, k, exact=True) * p**k * (1 - p)**(n - k) for k in range(n+1)]
