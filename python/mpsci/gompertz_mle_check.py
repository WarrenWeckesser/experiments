
from mpmath import mp
from mpsci.distributions import gompertz
import matplotlib.pyplot as plt


mp.dps = 100

x = [0.125, 0.25, 0.5, 0.625, 0.75, 1.25, 2.75, 3]

chat, scalehat = gompertz.mle(x)
print(f'{chat = }\n{scalehat = }')
nll_opt = gompertz.nll(x, chat, scalehat)

c = mp.linspace(0.95*chat, 1.05*chat, 300)
scale = mp.linspace(0.95*scalehat, 1.05*scalehat, 300)

yc = [gompertz.nll(x, t, scalehat) for t in c]
yscale = [gompertz.nll(x, chat, t) for t in scale]

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(c, yc)
ax1.plot(chat, nll_opt, 'k.')
ax1.grid(True)
ax2.plot(scale, yscale)
ax2.plot(scalehat, nll_opt, 'k.')
ax2.grid(True)

plt.show()
