
from mpmath import mp
from mpsci.distributions import kumaraswamy, Initial
import matplotlib.pyplot as plt


mp.dps = 100

x = [2e-6, 0.0001, 0.005, 0.0075, 0.125, 0.25, 0.5, 0.625, 0.75]

ahat, bhat = kumaraswamy.mle(x, a=Initial(0.25))
ahat = ahat.real
bhat = bhat.real
print(f'{ahat = }\n{bhat = }')
nll_opt = kumaraswamy.nll(x, ahat, bhat)

a = mp.linspace(0.95*ahat, 1.05*ahat, 300)
b = mp.linspace(0.95*bhat, 1.05*bhat, 300)

ya = [kumaraswamy.nll(x, t, bhat) for t in a]
yb = [kumaraswamy.nll(x, ahat, t) for t in b]

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(a, ya)
ax1.plot(ahat, nll_opt, 'k.')
ax1.grid(True)
ax2.plot(b, yb)
ax2.plot(bhat, nll_opt, 'k.')
ax2.grid(True)

plt.show()
