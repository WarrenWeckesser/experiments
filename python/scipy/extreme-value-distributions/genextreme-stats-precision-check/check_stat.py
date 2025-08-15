"""
Check the precision of the mean, variance, skewness or kurtosis statistics
of SciPy's implementation of the generalized extreme value distribution.
"""

import numpy as np
from scipy import stats
from mpmath import mp
from mpsci.distributions import genextreme
import matplotlib.pyplot as plt


mp.dps = 100


# statname can be "mean", "var", "skewness" or "kurtosis"
statname = "mean"

c = np.linspace(-0.15, 0.15, 1501)
relerror = []
mpsci_func = getattr(genextreme, statname)
scipy_vals = stats.genextreme.stats(c, moments=statname[0])
for c1, scipy_val1 in zip(c, scipy_vals):
    ref = mpsci_func(-c1)  # N.B. mpsci version uses opposite sign convention
    relerror.append(float(abs((scipy_val1 - ref))/ref))

relerror = np.array(relerror)

plt.plot(c, relerror)
plt.grid()
plt.xlabel('c')
plt.ylabel(f'relative error of the {statname}')
plt.semilogy()
plt.show()
