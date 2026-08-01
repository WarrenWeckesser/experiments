
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# from scipy.special import logit
from scipy.special._special_ufuncs import _tukeylambda_invcdf
from mpmath import mp
from mpsci.distributions import tukeylambda


def relerr(x, ref):
    if np.isinf(x) and float(ref) == x:
        return 0.0
    if ref == 0 or mp.isinf(ref):
        if x == ref:
            return 0.0
        else:
            return 1.0
    else:
        result = abs((x - ref)/ref)
        if mp.isinf(result):
            print(f"inf!\n{x = }\n{ref = }\n{float(ref) = }")
        return result


mp.dps = 400

lam = -216.75
# p = np.linspace(0.500000005, 0.98, 8000)
p = np.linspace(2e-16, 0.99999999999, 20000)
#x = stats.tukeylambda.ppf(p, lam)
x = _tukeylambda_invcdf(p, lam)
xmp = [tukeylambda.invcdf(p1, lam) for p1 in p]
re = [float(relerr(x1, x1mp)) for x1, x1mp in zip(x, xmp)]

print(np.max(re), np.mean(re))

plt.plot(p, re, '.', alpha=0.2, label='stats.tukeylambda.ppf')

plt.legend(framealpha=1, shadow=True)
plt.xlabel('p')
plt.ylabel('relative error')
plt.grid(True)
plt.semilogy()
plt.title(rf'$\lambda$ = {lam}')
plt.show()
