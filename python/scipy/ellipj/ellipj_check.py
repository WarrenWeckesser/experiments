
import mpmath
import numpy as np
from scipy.special import ellipj
from scipy.integrate import quad
from scipy.optimize import fsolve


def F(phi, k):
    return quad(lambda t: 1 / np.sqrt(1 - k**2*np.sin(t)**2), 0, phi,
                limit=250, epsabs=5e-15)[0]


def F1m(phi, eps):
    return quad(lambda t: 1 / np.sqrt(np.cos(t)**2 +
                                      eps*(2 - eps) * np.sin(t) ** 2),
                0, phi,
                limit=250, epsabs=5e-15)[0]


def ellipj_quad(u, k):
    phi = fsolve(lambda t: F(t, k) - u, 1.0, xtol=1e-12).item()
    sinphi = np.sin(phi)
    return sinphi, np.cos(phi), np.sqrt(1 - k**2*sinphi**2), phi


def ellipj1m_quad(u, eps):
    phi = fsolve(lambda t: F1m(t, eps) - u, 1.0, xtol=1e-12).item()
    sinphi = np.sin(phi)
    return sinphi, np.cos(phi), np.sqrt(1 - k**2*sinphi**2), phi


k = 0.99999999997
u = 50

m = k**2

sn, cn, dn, ph = ellipj(u, m)
snq, cnq, dnq, phq = ellipj_quad(u, k)

print()
print(f'k = {k}')
print(f'm = k**2 = {m}')
print(f'u = {u}')

print(f'         {"sn ":>25} {"cn ":>25} {"dn ":>25} {"phi ":>25}')
print(f'ellipj: {sn:25.17e} {cn:25.17e} {dn:25.17e} {ph:25.17e}')
print(f'quad:   {snq:25.17e} {cnq:25.17e} {dnq:25.17e} {phq:25.17e}')

eps = 1 - k
snq1m, cnq1m, dnq1m, phq1m = ellipj1m_quad(u, eps)

print(f'quad1m: {snq1m:25.17e} {cnq1m:25.17e} {dnq1m:25.17e} {phq1m:25.17e}')

mpmath.mp.dps = 100
snmp = float(mpmath.ellipfun('sn', u, m))
cnmp = float(mpmath.ellipfun('cn', u, m))
dnmp = float(mpmath.ellipfun('dn', u, m))

print(f'mpmath: {snmp:25.17e} {cnmp:25.17e} {dnmp:25.17e}')
