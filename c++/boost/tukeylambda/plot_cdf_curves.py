
import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp
from mpsci.distributions import tukeylambda


lam = -0.5
a = 1/lam
b = -a
R = -5*a
x = mp.linspace(-R, R, 500)
cdf = [tukeylambda.cdf(t, lam) for t in x]

tangent_bounds = [-2**(1-lam), 2**(1-lam)]
xd = mp.linspace(*tangent_bounds, 50)
pd = [0.5 + 2**(lam-2)*t for t in xd]

c = -2**(-lam)/lam
xcpos = mp.linspace(c, R, 500)
ycpos = [-mp.powm1(-lam*t, 1/lam) for t in xcpos]
xcneg = mp.linspace(-R, -c, 500)
ycneg = [mp.power(lam*t, 1/lam) for t in xcneg]

T = 1/(2*((-lam)**(1/lam - 1) + (-lam)**(1/lam)))**lam
unT = (-lam*T)**(1/lam)
vpT = 1 - unT

fig, ax = plt.subplots()

plt.plot(x, cdf)

tangcolor = '#5080A0'
plt.plot(xd, pd, '--', alpha=0.5, color=tangcolor)
plt.hlines(1, tangent_bounds[1], R, linestyles='--', alpha=0.5, color=tangcolor)
plt.hlines(0, -R, tangent_bounds[0], linestyles='--', alpha=0.5, color=tangcolor)

plt.plot(tangent_bounds, [0, 1], '.', color=tangcolor, alpha=0.5)

plt.plot(xcpos, ycpos, '-.', color='k', alpha=0.6)
plt.plot(xcneg, ycneg, '-.', color='k', alpha=0.6)

plt.plot(c, 0.5, '.', color='k', alpha=0.6)
plt.plot(-c, 0.5, '.', color='k', alpha=0.6)

plt.hlines(0.5, -c, 0, linestyles='-.', alpha=0.6, color='k')
plt.hlines(0.5, 0, c, linestyles='-.', alpha=0.6, color='k')


plt.plot(-T, unT, '.', color='red', alpha=0.4)
plt.plot(T, vpT, '.', color='red', alpha=0.4)
plt.plot([-T, T], [unT, vpT], ':', alpha=0.4, color='red')

plt.grid(alpha=0.5)
plt.xlabel("x")
plt.title(r"Tukey lambda CDF bounding curves, $\lambda < 0$")

ax.annotate(r"$\frac{1}{2} + 2^{\lambda-2}x$",
            xy=(xd[40], pd[40]), xycoords='data',
            xytext=(-30, 15), textcoords='offset points',
            arrowprops=dict(arrowstyle='->',
                            facecolor='black'),
            horizontalalignment='right', verticalalignment='center')

ax.annotate(r"$\left(\frac{-2^{-\lambda}}{\lambda}, \frac{1}{2}\right)$",
            xy=(c, 0.5), xycoords='data',
            xytext=(5, 0), textcoords='offset points',
            horizontalalignment='left', verticalalignment='center')

ax.annotate(r"$\left(\frac{2^{-\lambda}}{\lambda}, \frac{1}{2}\right)$",
            xy=(-c, 0.5), xycoords='data',
            xytext=(-5, 0), textcoords='offset points',
            horizontalalignment='right', verticalalignment='center')

ax.annotate(r"$(2^{1 - \lambda}, 1)$",
            xy=(2**(1 - lam), 1), xycoords='data',
            xytext=(-9, -2), textcoords='offset points',
            horizontalalignment='right', verticalalignment='center')

ax.annotate(r"$(-2^{1 - \lambda}, 0)$",
            xy=(-2**(1 - lam), 0), xycoords='data',
            xytext=(9, 5), textcoords='offset points',
            horizontalalignment='left', verticalalignment='center')

ax.annotate(r"$1 - (-\lambda x)^{1/\lambda}$",
            xy=(xcpos[300], ycpos[300]), xycoords='data',
            xytext=(0, -25), textcoords='offset points',
            arrowprops=dict(arrowstyle='->',
                            facecolor='black'),
            horizontalalignment='left', verticalalignment='top')

ax.annotate(r"$(\lambda x)^{1/\lambda}$",
            xy=(xcneg[200], ycneg[200]), xycoords='data',
            xytext=(-15, 15), textcoords='offset points',
            arrowprops=dict(arrowstyle='->',
                            facecolor='black'),
            horizontalalignment='right', verticalalignment='bottom')

plt.savefig('cdf_curves.svg')
plt.show()
