import numpy as np
from mpmath import mp
import matplotlib.pyplot as plt


mp.dps = 200


def logit_mp(p):
    p = mp.mpf(p)
    return float(mp.log(p/(1 - p)))


p, qb, qr, qs = np.loadtxt('out', unpack=True)

q = np.array([logit_mp(t) for t in p])

e1 = np.abs((qb - q)/q)
e2 = np.abs((qr - q)/q)
e3 = np.abs((qs - q)/q)

plt.plot(p, e1, label='boost logistic quantile\n(no promotion)', alpha=0.5)
plt.plot(p, e2, '--', label='atanh', alpha=0.5)
plt.plot(p, e3, label='scipy', alpha=0.5)

plt.xlabel('p')
plt.ylabel('logit(p)')
plt.legend()
plt.grid()
plt.show()
