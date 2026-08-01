
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logit
from mpmath import mp
from mpsci.distributions import tukeylambda


def relerr(x, ref):
    if ref == 0 or mp.isinf(ref):
        if x == ref:
            return 0.0
        else:
            return 1.0
    else:
        return abs((x - ref)/ref)


def try_this_save(p, lam):
    # if p > 0.425:
    if lam > 1:
        if p > 0.425:
            x = -np.pow(p, lam)/lam * np.expm1(-lam * logit(p))
        else:
            x = np.exp(lam * np.log1p(-p)) * np.expm1(lam  * logit(p)) / lam
    elif lam > 0.125:
        x = -np.pow(p, lam)/lam * np.expm1(-lam * logit(p))
    elif lam > 0.0:
        x = np.exp(lam * np.log1p(-p)) * np.expm1(lam  * logit(p)) / lam
    elif lam == 0:
        x = logit(p)
    else:
        # lam < 0
        if p > 0.5:

            # x = -np.pow(p, lam)/lam * np.expm1(-lam * logit(p))
            x = np.pow(1 - p, lam)/lam * np.expm1(lam * logit(p))

            ## x = np.exp(lam * np.log1p(-p)) * np.expm1(lam  * logit(p)) / lam
            #x = -np.exp(lam * np.log(p)) * np.expm1(lam  * -logit(p)) / lam

        else:
            x = -np.pow(p, lam)/lam * np.expm1(-lam * logit(p))
    return x


def try_this(p, lam):
    # Special cases...
    if lam == 2:
        return p - 0.5
    if lam == 1:
        return 2*p - 1
    if lam == 0:
        return logit(p)

    # General case: the formulas used here are all mathematically equivalent.
    # The robustnes of the floating point numerical calculation depends on the
    # values of the parameters.

    if lam > 1.5:
        if p > 0.4:
            x = -np.pow(p, lam)/lam * np.expm1(-lam * logit(p))
        else:
            x = np.exp(lam * np.log1p(-p)) / lam * np.expm1(lam  * logit(p))
            # x = np.pow(1 - p, lam) / lam * np.expm1(lam  * logit(p))

    elif lam > 1e-19:
        # good for 1e-19 <= lam <= 1.5
        # (but boundary cutoffs are a bit fuzzy)
        if p < 0.5:
            x = np.pow(1 - p, lam)/lam * np.expm1(lam * logit(p))
        else:
            x = -np.pow(p, lam)/lam * np.expm1(-lam * logit(p))

    elif lam > 0.0:
        if p < 0.4 or p > 0.6:
            # Use the "regular" formula
            x = stats.tukeylambda.ppf(p, lam)
        elif p < 0.5:
            # 0.4 <= p < 0.5
            x = np.pow(1 - p, lam)/lam * np.expm1(lam * logit(p))
        else:
            # 0.5 <= p <= 0.6
            x = -np.pow(p, lam)/lam * np.expm1(-lam * logit(p))

    elif lam > -1e-20:
        if p < 0.4 or p > 0.6:
            # Use the "regular" formula
            x = stats.tukeylambda.ppf(p, lam)
        elif p > 0.5:  # actually 0.5 < p <= 0.6
            x = np.pow(1 - p, lam)/lam * np.expm1(lam * logit(p))
        else:
            # 0.4 < p <= 0.5
            x = -np.pow(p, lam)/lam * np.expm1(-lam * logit(p))
    else:
        # lam < -1e-20
        if p > 0.5:
            x = np.pow(1 - p, lam)/lam * np.expm1(lam * logit(p))
        else:
            x = -np.pow(p, lam)/lam * np.expm1(-lam * logit(p))
    return x


try_this = np.vectorize(try_this, otypes=['d'])

mp.dps = 400

lam = 125.0
# p = np.linspace(0.500000005, 0.98, 8000)
p = np.linspace(2e-16, 0.99999999999, 20000)
x = stats.tukeylambda.ppf(p, lam)
#xa = -np.pow(p, lam)/lam * np.expm1(-lam * logit(p))
#xb = np.exp(lam * np.log1p(-p)) * np.expm1(lam  * logit(p)) / lam
xt = try_this(p, lam)
xmp = [tukeylambda.invcdf(p1, lam) for p1 in p]
re = [float(relerr(x1, x1mp)) for x1, x1mp in zip(x, xmp)]
#rea = [float(relerr(x1, x1mp)) for x1, x1mp in zip(xa, xmp)]
#reb = [float(relerr(x1, x1mp)) for x1, x1mp in zip(xb, xmp)]
ret = [float(relerr(x1, x1mp)) for x1, x1mp in zip(xt, xmp)]

print(np.max(re), np.mean(re))
#print(np.max(rea), np.mean(rea))
#print(np.max(reb), np.mean(reb))
print(np.max(ret), np.mean(ret))

plt.plot(p, re, '.', alpha=0.2, label='stats.tukeylambda.ppf')
#plt.plot(p, rea, '.', alpha=0.25, label='(a)')
#plt.plot(p, reb, '.', alpha=0.125, label='(b)')
plt.plot(p, ret, 'm.', alpha=0.125, label='(t)')

plt.legend(framealpha=1, shadow=True)
plt.xlabel('p')
plt.ylabel('relative error')
plt.grid(True)
plt.semilogy()
plt.title(rf'$\lambda$ = {lam}')
plt.show()
