"""
Integrate the Frenet-Serret formulas to construct a curve,
given:
* curvature and torsion as functions of arclength
* initial point, tangent and normal vectors
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt


# In the following, `s` is arclength.

def curve_eqns(s, z, curvature, torsion):
    t = z[3:6]
    n = z[6:9]
    b = z[9:]
    kappa = curvature(s)
    tau = torsion(s)
    dzds = np.concatenate((t, kappa*n, -kappa*t + tau*b, -tau*n))
    return dzds


def curvature(s):
    return 1.0


def torsion(s):
    return 0.1 + 1/(s**2 + 1)


# Initial values...
# Starting point:
x0 = [0, 0, 0]
# Tangent (a unit vector):
t0 = [0, 1, 0]
# Normal (a unit vector orthogonal to t0):
n0 = [0, 0, 1]
# Binornmal (completes the initial 3-d Frenet-Serret frame):
b0 = np.cross(t0, n0)
z0 = np.concatenate((x0, t0, n0, b0))

s = np.linspace(-50, 50, 2000)
use_odeint = True
rtol = 5e-13
if use_odeint:
    sol = odeint(curve_eqns, z0, s, args=(curvature, torsion),
                 rtol=rtol, tfirst=True)
else:
    result = solve_ivp(curve_eqns, [s[0], s[-1]], z0, t_eval=s,
                       args=(curvature, torsion),
                       rtol=rtol, method='Radau')
    sol = result.y.T

ff = sol[-1, 3:].reshape((3, 3)).T
print("ff (the final Frenet-Serret frame):")
print(ff)
print()
print("ff @ ff.T (ideally this is the identity matrix):")
print(ff @ ff.T)
print()
print("1 - det(ff) =", 1 - np.linalg.det(ff),
      " (ideally this is 0)")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(sol[:, 0], sol[:, 1], sol[:, 2])
# Simulate equal aspect ratio:
ptp = sol[:, :3].ptp(axis=0)
ax.set_box_aspect(ptp)

plt.show()
