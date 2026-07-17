# Scratch work for checking relative errors of some moment calculations
# for the generalized extreme value distribution.

import numpy as np
from mpmath import mp
from scipy.special import gamma


npnames = dict(vartype=float, nan=np.nan, pi=np.pi, gamma=gamma)
mpnames = dict(vartype=mp.mpf, nan=mp.nan, pi=mp.pi, gamma=mp.gamma)


def var_term(xi, namespace):
    vartype = namespace['vartype']
    nan = namespace['nan']
    pi = namespace['pi']
    gamma = namespace['gamma']
    y = []
    for xi1 in xi:
        xi1 = vartype(xi1)
        if xi1 >= 0.5:
            y.append(nan)
        elif xi1 == 0:
            y.append(pi**2/6)
        else:
            g1 = gamma(1 - xi1)
            g2 = gamma(1 - 2*xi1)
            y.append((g2 - g1**2)/xi1**2)
    return y


@mp.workdps(80)
def var_term_relerror(xi, y):
    relerrs = []
    ref = var_term(xi, namespace=mpnames)
    for xi1, y1, ref1 in zip(xi, y, ref):
        if ref1 == 0:
            relerrs.append(0.0)
        else:
            relerrs.append(float(abs(y1 - ref1)/ref1))
    return np.array(relerrs)


def skewness_term(xi):
    g1 = gamma(1 - xi)
    g2 = gamma(1 - 2*xi)
    g3 = gamma(1 - 3*xi)
    return g3 - 3*g2*g1 + 2*g1**3


@mp.workdps(80)
def skewness_term_relerror(xi, y):
    relerrs = []
    for xi1, y1 in zip(xi, y):
        xi1mp = mp.mpf(float(xi1))
        ymp = mp.mpf(float(y1))
        g1 = mp.gamma(1 - xi1mp)
        g2 = mp.gamma(1 - 2*xi1mp)
        g3 = mp.gamma(1 - 3*xi1mp)
        ref = g3 - 3*g2*g1 + 2*g1**3
        if ref == 0:
            relerrs.append(0.0)
        else:
            relerrs.append(float(abs((ymp - ref)/ref)))
    return np.array(relerrs)


def kurtosis_term(xi):
    g1 = gamma(1 - xi)
    g2 = gamma(1 - 2*xi)
    g3 = gamma(1 - 3*xi)
    g4 = gamma(1 - 4*xi)
    return g4 - 4*g3*g1 + 6*g1**2*g2 - 3*g1**4


@mp.workdps(80)
def kurtosis_term_relerror(xi, y):
    relerrs = []
    for xi1, y1 in zip(xi, y):
        xi1mp = mp.mpf(float(xi1))
        ymp = mp.mpf(float(y1))
        g1 = mp.gamma(1 - xi1mp)
        g2 = mp.gamma(1 - 2*xi1mp)
        g3 = mp.gamma(1 - 3*xi1mp)
        g4 = mp.gamma(1 - 4*xi1mp)
        ref = g4 - 4*g3*g1 + 6*g1**2*g2 - 3*g1**4
        if ref == 0:
            relerrs.append(0.0)
        else:
            relerrs.append(float(abs((ymp - ref)/ref)))
    return np.array(relerrs)


if __name__ == "__main__":
    xi = np.array([-100.0, -10.0, -1.0, -0.5, 0.0, 0.25, 0.499])
    v = var_term(xi, npnames)
    re = var_term_relerror(xi, v)
    widths = [13, 15, 10]
    print("                      computed    relative")
    print("     xi               variance       error")
    print("-------------     ------------   ---------")
    for xi1, v1, re1 in zip(xi, v, re):
        print(f"{xi1:{widths[0]}.6f}  {v1:{widths[1]}g}  {re1:{widths[2]}.3e}")
