"""
Integrate the Frenet-Serret formulas to construct a curve,
given:
* curvature and torsion as functions of arclength
* initial point, tangent and normal vectors
"""

# Copyright (c) 2022 Warren Weckesser
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt


# In the following, `s` is arclength.

def curve_eqns(s, z, curvature, torsion):
    # t and n are the tangent and normal vectors
    t = z[3:6]
    n = z[6:9]
    # b is the binormal vector
    b = np.cross(t, n)
    kappa = curvature(s)
    tau = torsion(s)
    dzds = np.concatenate((t, kappa*n, -kappa*t + tau*b))
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

z0 = np.concatenate((x0, t0, n0))

s = np.linspace(-50, 50, 2000)
rtol = 5e-13
use_odeint = False
if use_odeint:
    sol = odeint(curve_eqns, z0, s, args=(curvature, torsion),
                 rtol=rtol, tfirst=True)
else:
    result = solve_ivp(curve_eqns, [s[0], s[-1]], z0, t_eval=s,
                       args=(curvature, torsion),
                       rtol=rtol, method='Radau')
    sol = result.y.T

tf = sol[-1, 3:6]
nf = sol[-1, 6:9]
bf = np.cross(tf, nf)
# The final Frenet-Serret frame:
ff = np.column_stack((tf, nf, bf))

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
ptp = np.ptp(sol[:, :3], axis=0)
ax.set_box_aspect(ptp)

# plt.show()
plt.savefig('curve.svg')
