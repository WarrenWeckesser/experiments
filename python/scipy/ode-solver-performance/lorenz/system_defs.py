import numpy as np
from scipy.fftpack import diff as psdiff


def lorenz_sys(t,  u, sigma, rho, beta):
    x, y, z = u
    return np.array([-sigma*(x - y), -y + x*rho - x*z, x*y - beta*z])


def lorenz_jac(t, u, sigma, rho, beta):
    x, y, z = u

    jac = np.zeros((3, 3))
    jac[0, 0] = -sigma
    jac[0, 1] = sigma
    jac[1, 0] = rho - z
    jac[1, 1] = -1
    jac[1, 2] = -x
    jac[2, 0] = y
    jac[2, 1] = x
    jac[2, 2] = -beta

    return jac


def rossler_sys(t, u, p):
    x, y, z = u
    a, b, c = p
    return np.array([-z - y, x + a*y, z*(x - c) + b])


def rossler_jac(t, u, p):
    x, y, z = u
    a, b, c = p

    jac = np.zeros((3, 3))
    jac[0, 1] = -1
    jac[0, 2] = -1
    jac[1, 0] = 1
    jac[1, 1] = a
    jac[2, 0] = z
    jac[2, 2] = x - c

    return jac


def kdv(t, u, L):
    """Differential equations for the KdV equation, discretized in x."""
    # Compute the x derivatives using the pseudo-spectral method.
    ux = psdiff(u, period=L)
    uxxx = psdiff(u, period=L, order=3)

    # Compute du/dt.
    dudt = -6*u*ux - uxxx

    return dudt
