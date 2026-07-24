from mpmath import mp
import numpy as np
from scipy.fft import dct
import matplotlib.pyplot as plt
from chebstuff import chebyshev_coefficients, chebyshev_eval, chebyshev_lobatto_sample



def mp_xlogy(x, y):
    if x == 0:
        return mp.zero
    return x * mp.log(y)


def relerr(y, yref):
    re = []
    for y1, yref1 in zip(y, yref):
        if yref1 == 0:
            if y1 == 0:
                re.append(0.0)
            else:
                re.append(np.inf)
        else:
            re.append(abs((y1 - yref1) / yref1))
    return np.array(re)


def gompertz_variance(eta):
    eta = mp.mpf(eta)
    if eta == 0:
        return mp.pi**2/6
    e1 = mp.expint(1, eta)
    mean = mp.exp(eta) * e1
    # mu_2': second noncentral moment
    mu2p = mp.exp(eta)*(-2*eta*mp.hyper([1, 1, 1], [2, 2, 2], -eta) + (mp.euler + mp.log(eta))**2 + mp.pi**2/6)
    return mu2p - mean**2

def gompertz_variance_small_eta_old(eta):
    eta = mp.mpf(eta)
    if eta == 0:
        return mp.pi**2/6
    return (mp.pi**2/6 + eta * (-mp.log(eta)**2 + (2 - 2*mp.euler)*mp.log(eta) - 2 - mp.euler**2
            + 2*mp.euler + mp.pi**2/6))

def gompertz_variance_small_eta(eta):
    eta = mp.mpf(eta)
    if eta == 0:
        return mp.pi**2/6
    log = mp.log
    EulerGamma = mp.euler
    pi = mp.pi
    return eta**2*(-3*log(eta)**2/2 - 3*EulerGamma*log(eta) + 4*log(eta) - 7/4 - 3*EulerGamma**2/2 + pi**2/12 + 4*EulerGamma - log(eta)**2/(log(eta)**2 + 2*EulerGamma*log(eta) + EulerGamma**2) - 2*EulerGamma*log(eta)/(log(eta)**2 + 2*EulerGamma*log(eta) + EulerGamma**2) - EulerGamma**2/(log(eta)**2 + 2*EulerGamma*log(eta) + EulerGamma**2) + 2*log(eta)**2/(-4*log(eta) - 4*EulerGamma) + 4*EulerGamma*log(eta)/(-4*log(eta) - 4*EulerGamma) + 2*EulerGamma**2/(-4*log(eta) - 4*EulerGamma)) + eta*(-log(eta)**2 - 2*EulerGamma*log(eta) + 2*log(eta) - 2 - EulerGamma**2 + 2*EulerGamma + pi**2/6) + pi**2/6

def gompertz_variance_small_eta_float(eta):
    if eta == 0:
        return np.pi**2/6
    log = np.log
    EulerGamma = np.euler_gamma
    pi = np.pi
    return eta**2*(-3*log(eta)**2/2 - 3*EulerGamma*log(eta) + 4*log(eta) - 7/4 - 3*EulerGamma**2/2 + pi**2/12 + 4*EulerGamma - log(eta)**2/(log(eta)**2 + 2*EulerGamma*log(eta) + EulerGamma**2) - 2*EulerGamma*log(eta)/(log(eta)**2 + 2*EulerGamma*log(eta) + EulerGamma**2) - EulerGamma**2/(log(eta)**2 + 2*EulerGamma*log(eta) + EulerGamma**2) + 2*log(eta)**2/(-4*log(eta) - 4*EulerGamma) + 4*EulerGamma*log(eta)/(-4*log(eta) - 4*EulerGamma) + 2*EulerGamma**2/(-4*log(eta) - 4*EulerGamma)) + eta*(-log(eta)**2 - 2*EulerGamma*log(eta) + 2*log(eta) - 2 - EulerGamma**2 + 2*EulerGamma + pi**2/6) + pi**2/6


def gompertz_small_eta_variance_factor(eta):
    eta = mp.mpf(eta)
    if eta == 0:
        return mp.zero
    e1 = mp.expint(1, eta)
    mean = mp.exp(eta) * e1
    # mu_2': second noncentral moment
    mu2p = mp.exp(eta)*(-2*eta*mp.hyper([1, 1, 1], [2, 2, 2], -eta) + (mp.euler + mp.log(eta))**2 + mp.pi**2/6)
    var = mu2p - mean**2
    f = var - mp.pi**2/6 - mp_xlogy(eta, eta) - (2 - mp.pi**2/6)*eta
    return f


def gompertz_variance_using_quad(eta):
    if eta == 0:
        return mp.pi**2/6
    e1 = mp.expint(1, eta)
    mean = mp.exp(eta) * e1
    # mu_2': second noncentral moment
    mu2p = mp.exp(eta)*mp.quad(lambda u: mp.exp(-u) * mp.log(u/eta)**2, [eta, mp.inf])
    return mu2p - mean**2


def foo(eta):
    with mp.extraprec(mp.prec):
        return gompertz_variance(eta) - gompertz_variance_small_eta(eta) 

# Need to be smarter near eta = 0.  Maybe figure out some terms in the
# asymptotic expansion at eta = 0, and use the Chebyshev polynomial
# on the different between the truncated asymptotic and the true value.
# foo(eta) above is part of some scratch work for this.
# See the file gompertz_variance_small_eta_sympy.py for an expansion
# at eta = 0.

# a = 2.7e-7, b = 3.5e-6, N = 32, max_relerr = 8.449128503789453e-16
# a = 3.5e-6, b = 3.5e-5, N = 32, max_relerr = 4.807307126898555e-16
# a = 3.5e-5, b = 0.0003, N = 32, max_relerr = 5.835617438564162e-16
# a = 0.0003, b = 0.0023, N = 32, max_relerr = 8.728465433660703e-16
# a = 0.0023, b = 0.015,  N = 32, max_relerr = 5.213419544261299e-16
# a = 0.015,  b = 0.085,  N = 32, max_relerr = 4.481786775401879e-16
# a = 0.085,  b = 0.43,   N = 32, max_relerr = 4.1099054077508636e-16
# a = 0.43,   b = 2,      N = 32, max_relerr = 7.884578323709884e-16
# a = 2,      b = 7,      N = 32, max_relerr = 5.554110113503729e-16
# a = 7,      b = 23,     N = 32, max_relerr = 5.160561625266786e-16
# a = 23,     b = 75,     N = 32, max_relerr = 7.726090061807916e-16
# a = 75,     b = 250,    N = 32, max_relerr = 8.951516111483102e-16
# a = 250,    b = 800,    N = 32, max_relerr = 6.098706118755169e-16

N = 32
# func = foo
func = gompertz_variance
a = 2
b = 7

mp.dps = 800

# t, x, y = chebyshev_lobatto_sample(mp, gompertz_variance, a, b, N)
t, x, y = chebyshev_lobatto_sample(mp, func, a, b, N)
mp_coefs = chebyshev_coefficients(mp, y, N)
coefs = [float(q) for q in mp_coefs]
# print(f"coefs: {coefs}")

xx = np.linspace(a, b, 2500)
# yy = np.array([float(gompertz_variance(h)) for h in xx])
yy = np.array([float(func(h)) for h in xx])

tt = (2.0*xx - (a + b)) / (b - a)
yyi = [chebyshev_eval(t1, coefs) for t1 in tt]
# yy_ref = [gompertz_variance(x1) for x1 in xx]
yy_ref = [func(x1) for x1 in xx]

# Max relative error in the interpolated values...
re = relerr(yyi, yy_ref)
max_relerr = float(max(re))
print(f"{max_relerr = }")

plt.plot(xx, yy, label="Original function")
plt.plot(xx, yyi, linewidth=5, alpha=0.5, label="Chebyshev polynomial approximation")
plt.legend(framealpha=1, shadow=True)
plt.grid()
plt.show()
