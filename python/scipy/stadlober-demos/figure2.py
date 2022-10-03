"""
Create a plot similar to Figure 2 from [1]_.

.. [1] Ernst Stadlober, The ratio of uniforms approach for
   generating discrete random variates, Journal of Computational
   and Applied Mathematices 31 (1990) 181-189

"""

import math
import numpy as np
from scipy.stats import hypergeom
import matplotlib.pyplot as plt


def hypergeom_meanvar(ngood, nbad, nsample):
    total = ngood + nbad
    mu = nsample * ngood / total
    var = mu * (nbad/total) * (total - nsample)/(total - 1)
    return mu, var


def tablemtn(x, loc, scale):
    y = np.zeros(x.shape)
    z = (x - loc)/scale
    mask = (z >= -1) & (z <= 1)
    y[mask] = 0.25/scale
    y[~mask] = (0.25*scale / (x[~mask] - loc)**2)
    return y


# Use total=256, ngood=64, nsample=16 to recreate Figure 2.
total = 125
ngood = 25
nsample = 50

nbad = total - ngood

if nsample > total/2:
    raise ValueError('nsample must not be greater than total/2')
if ngood > total/2:
    raise ValueError('ngood must not be greater than total/2')

mu, var = hypergeom_meanvar(ngood, nbad, nsample)

k = np.arange(min(nsample, ngood) + 1)
p = hypergeom.pmf(k, total, ngood, nsample)
p /= p.max()

a = mu + 0.5
q = 1 - ngood/total

z = a - math.sqrt(2*a*q*(1 - nsample/total))

# Either kstar_floor or kstar_ceil is the optimal kstar
# (see Stadlober's thesis, page 82, between equations 5.18 and 5.19).
kstar_floor = math.floor(z)
kstar_ceil = math.ceil(z)
print("z =", z, "   kstar_floor =", kstar_floor, "   kstar_ceil =", kstar_ceil)
fstar_floor = p[kstar_floor]
fstar_ceil = p[kstar_ceil]
sstar_floor = (a - kstar_floor)*math.sqrt(fstar_floor)
sstar_ceil = (a - kstar_ceil)*math.sqrt(fstar_ceil)
print("sstar_floor =", sstar_floor, "   sstar_ceil =", sstar_ceil)

h1 = (4*sstar_ceil)*tablemtn(np.array(kstar_floor), loc=a, scale=sstar_ceil)
# h2 = (4*sstar_floor)*tablemtn(np.array(kstar_ceil), loc=a, scale=sstar_floor)
if h1 < p[kstar_floor]:
    kstar = kstar_floor
else:
    kstar = kstar_ceil
fstar = p[kstar]
sstar = (a - kstar)*math.sqrt(fstar)
print("sstar =", sstar)

# sstar is the scale of the table mountain function whose graph just touches
# the PMF (at kstar) of the probability distribution.
# s_hat is an overestimate of that value.  The advantage of using s_hat is that
# it is easier to compute.
c = np.sqrt(var + 0.5)
D1 = 1.7155277699214135
D2 = 0.8989161620588988
s_hat = 0.5*(D1*c + D2)
print("s_hat =", s_hat)

xx = np.linspace(-2, min(nsample, ngood)+1, 12000)

loc = a
yy = (4*sstar)*tablemtn(xx, loc=a, scale=sstar)

#yy_floor = (4*sstar_floor)*tablemtn(xx, loc=a, scale=sstar_floor)
#yy_ceil = (4*sstar_ceil)*tablemtn(xx, loc=a, scale=sstar_ceil)

yy_hat = (4*s_hat)*tablemtn(xx, loc=a, scale=s_hat)

plt.bar(k, p, align='edge', width=1, alpha=0.3, edgecolor='k')
plt.plot(xx, yy, 'k', linewidth=1, label="Using sstar")
#plt.plot(xx, yy_floor, 'k', linewidth=1)
#plt.plot(xx, yy_ceil, 'k--', linewidth=1)
plt.plot(xx, yy_hat, 'g', linewidth=1, alpha=0.5, label="Using s_hat")

plt.plot(kstar, p[kstar], 'k.')
#plt.plot(kstar_floor, p[kstar_floor], 'k.')
#plt.plot(kstar_ceil, p[kstar_ceil], 'k.')

#plt.plot(kstar_floor, (4*sstar_ceil)*tablemtn(np.array(kstar_floor), loc=a, scale=sstar_ceil), 'r.')
#plt.plot(kstar_ceil, (4*sstar_floor)*tablemtn(np.array(kstar_ceil), loc=a, scale=sstar_floor), 'r.')

#if (4*sstar_ceil)*tablemtn(np.array(kstar_floor), loc=a, scale=sstar_ceil) < p[kstar_floor]:
#    print("kstar_ceil is bad!")
#if (4*sstar_floor)*tablemtn(np.array(kstar_ceil), loc=a, scale=sstar_floor) < p[kstar_ceil]:
#    print("kstar_floor is bad!")

plt.xticks([t for t in plt.xticks()[0] if t == int(t)])
plt.grid(True)
plt.legend(shadow=True, framealpha=1)
plt.show()
