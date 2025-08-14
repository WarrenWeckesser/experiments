import numpy as np
from mpmath import mp
import matplotlib.pyplot as plt


mp.dps = 200


def logit_mp(p):
    p = mp.mpf(p)
    return float(mp.log(p/(1 - p)))


p, qb, qr, qs, qw = np.loadtxt('out', unpack=True)

q = np.array([logit_mp(t) for t in p])

e1 = np.abs((qb - q)/q)
e2 = np.abs((qr - q)/q)
e3 = np.abs((qs - q)/q)
e4 = np.abs((qw - q)/q)

plt.plot(p, e1, label='boost/math logistic quantile', alpha=0.6)
plt.plot(p, e2, '--', label='2*atanh(2*p-1)', alpha=0.6)
# plt.plot(p, e3, label='scipy', alpha=0.6)
plt.plot(p, e4, label='proposed', alpha=0.6)

plt.xlabel('p')
plt.ylabel('logit(p) relative error')
plt.legend(framealpha=1, shadow=True)
plt.semilogy()
plt.grid()
plt.show()
