# Scratch work

import numpy as np
from scipy.special import betaincc, betainc, beta, betaln, hyp2f1
from mpmath import mp


def cdf(k, n, p):
    return betainc(n - k, 1 + k, 1 - p)


def cdf2(k, n, p):
    a = n - k
    b = 1 + k
    x = 1 - p
    return x**a * (1 - x)**b * hyp2f1(a + b, 1, a + 1, x) / (a*beta(a, b))


def binomial_logcdf(k, n, p):
    a = n - k
    b = 1 + k
    x = 1 - p
    return (-np.log(a) + a*np.log(x) + b*np.log1p(-x) + np.log(hyp2f1(a + b, 1, a + 1, x))
            - betaln(a, b))
    

def sf2(k, n, p):
    a = n - k
    b = 1 + k
    x = p
    b, a = a, b
    return x**a * (1 - x)**b * hyp2f1(a + b, 1, a + 1, x) / (a*beta(a, b))


def binomial_logsf(k, n, p):
    a = 1 + k
    b = n - k
    x = p
    #return (-np.log(a) + a*np.log(x) + b*np.log1p(-x) + np.log(hyp2f1(a + b, 1, a + 1, x))
    #        - betaln(a, b))
    return float(mp.fsum([mp.mpf(-np.log(a)), mp.mpf(a*np.log(x)), mp.mpf(b*np.log1p(-x)),
                   mp.mpf(np.log(hyp2f1(a + b, 1, a + 1, x))), mp.mpf(-betaln(a, b))]))

def mp_binomial_logcdf(k, n, p):
    a = n - k
    b = 1 + k
    x = 1 - mp.mpf(p)
    return (-mp.log(a) + a*mp.log(x) + b*mp.log1p(-x) + mp.log(mp.hyp2f1(a + b, 1, a + 1, x))
            - mp.log(mp.beta(a, b)))

def mp_binomial_logsf(k, n, p):
    a = 1 + k
    b = n - k
    x = mp.mpf(p)
    return (-mp.log(a) + a*mp.log(x) + b*mp.log1p(-x) + mp.log(mp.hyp2f1(a + b, 1, a + 1, x))
            - mp.log(mp.beta(a, b)))
