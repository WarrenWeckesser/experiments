
from mpmath import mp
from mpsci.distributions import lognormal, Initial
import matplotlib.pyplot as plt


mp.dps = 100

x = [0.0001, 0.005, 0.0075, 0.125, 0.25, 0.5, 0.625, 0.75, 1.25, 2.75]

muhat, sigmahat = lognormal.mle(x)
print(f'{muhat = }\n{sigmahat = }')
nll_opt = lognormal.nll(x, muhat, sigmahat)

mu = mp.linspace(0.95*muhat, 1.05*muhat, 300)
sigma = mp.linspace(0.95*sigmahat, 1.05*sigmahat, 300)

ymu = [lognormal.nll(x, t, sigmahat) for t in mu]
ysigma = [lognormal.nll(x, muhat, t) for t in sigma]

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(mu, ymu)
ax1.plot(muhat, nll_opt, 'k.')
ax1.grid(True)
ax2.plot(sigma, ysigma)
ax2.plot(sigmahat, nll_opt, 'k.')
ax2.grid(True)

plt.show()
