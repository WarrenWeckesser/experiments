from timeit import timeit
import numpy as np
from scipy.integrate import odeint, solve_ivp
from system_defs import (lorenz_sys, lorenz_jac, rossler_sys, rossler_jac,
                          kdv)
import matplotlib.pyplot as plt


# Lorenz parameters
sigma = 10
rho = 28
beta = 10/3
# Initial condition
u0 = [4, 3, 5]
# Solver tolerances
reltol = 1e-9
abstol = 1e-12

tfinal = 50
t = np.linspace(0, tfinal, 3000)

sol1 = odeint(lorenz_sys, u0, t, args=(sigma, rho, beta), Dfun=lorenz_jac,
             rtol=reltol, atol=abstol, tfirst=True)

sol2 = solve_ivp(lorenz_sys, [t[0], t[-1]], u0, t_eval=t, method='LSODA', args=(sigma, rho, beta),
                 rtol=reltol, atol=abstol)

plt.plot(sol2.y[0], sol2.y[2], alpha=0.5)
plt.grid()
plt.show()
