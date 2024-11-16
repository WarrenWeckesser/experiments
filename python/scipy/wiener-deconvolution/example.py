import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt


th = np.linspace(0, 20, 1601)
h = np.exp(-th/1.25) - np.exp(-th/0.75)
h /= h.max()


n = 8000
x = np.random.normal(0, 0.002, size=n)
tvals = [800, 1000, 1100, 1700, 2800, 3200, 4800, 5200, 5400, 5500]
spikevals = [0.5, 1, 0.5, 0.5, 1, 1, 1, 1, 1, 0.75]
x[tvals] = spikevals

y = np.convolve(x, h, mode='full')
y += 0.0005*np.random.lognormal(-1, 1, size=len(y))
# y *= 0.25*np.random.lognormal(0, .0001, size=len(y))

# sn = 1000000
sn = np.inf
H = fft(h, n=len(y))
if sn == np.inf:
    G = H.conj()/(np.abs(H)**2)
else:
    G = H.conj()*sn/(np.abs(H)**2*sn + 1)

Y = fft(y)
Xhat = G*Y
xhat = ifft(Xhat).real

# plt.plot(x, '.', alpha=0.25, label='x')
plt.plot(y, 'g', linewidth=1, label='response signal')
plt.plot(np.arange(len(xhat)), xhat, 'r', alpha=0.5, linewidth=1,
         label='deconvolved signal')
plt.grid()
plt.legend()
plt.show()
