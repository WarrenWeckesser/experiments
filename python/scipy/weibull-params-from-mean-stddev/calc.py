
import math
import numpy as np

from scipy.special import gamma, gammaln
from scipy.optimize import fsolve
from scipy.stats import weibull_min

def func(params, mean, stddev):
    c, scale = params
    mu, sig2 = weibull_min.stats(c, scale=scale, moments='mv')
    return [mu - mean, sig2 - stddev**2]


def weibull_params(mn, sd, guess=[1.0, 1.0]):
    c0 = 1.27*(mn/sd)**2
    p, info, ier, msg = fsolve(func, guess, args=(mn, sd),
                               full_output=True)
    if ier != 1:
        raise RuntimeError(f'fsolve failed to find a solution: {msg}')
    return p


def check_weibull_params(sds):
    mn = 1
    guess = [1, 1]
    result = []
    for sd in sds:
        try:
            p = weibull_params(mn, sd, guess=guess)
        except Exception:
            p = None
        result.append(p)
    return result


def g(c):
    g1sq = gamma(1 + 1/c)**2
    return math.sqrt(g1sq / (gamma(1 + 2/c) - g1sq))

def h(c):
    r = np.exp(gammaln(2/c) - 2*gammaln(1/c))
    return np.sqrt(1/(2*c*r - 1))


def weibull_c_scale_from_mean_std(mean, std):
    c0 = 1.27*math.sqrt(mean/std)
    c, info, ier, msg = fsolve(lambda t: h(t) - (mean/std), c0, xtol=1e-10,
                               full_output=True)
    if ier != 1:
        raise RuntimeError(f'fsolve failed: {msg}')
    scale = mean / gamma(1 + 1/c)
    return c, scale
