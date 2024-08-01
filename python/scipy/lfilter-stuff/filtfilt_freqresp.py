
import numpy as np
from scipy.signal import filtfilt, butter, freqz, lfilter
import matplotlib.pyplot as plt


wa = 0.125
wb = 0.5
b, a = butter(4, (wa, wb), btype='bandpass')
w, h = freqz(b, a, worN=8000)


# Compute the impulse response of the forward-backward filter.
n = 501
u = np.zeros(n)
u[n//2] = 1
v1 = lfilter(b, a, u)
v2 = lfilter(b, a, v1[::-1])[::-1]
# v1 is the impulse response.
ww, hh = freqz(v2, 1, worN=8000)

plt.plot(w/np.pi, np.abs(h), label='Forward')
plt.plot(w/np.pi, np.abs(h)**2, label='Forward-Backward')
plt.plot(ww/np.pi, np.abs(hh), label='Forward-Backward\n(alternate)',
            linewidth=4, alpha=0.2)
plt.ylim(0, 1.05)
plt.grid(True)
plt.axvline(wa, color='k', alpha=0.25)
plt.axvline(wb, color='k', alpha=0.25)
plt.title('Butterworth Bandpass Filter')
plt.legend()
plt.show()
