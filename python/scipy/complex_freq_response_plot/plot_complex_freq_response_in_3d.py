import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, remez


taps = remez(31, [0, 0.3, 0.4, 1], [1, 0], fs=2)

w, h = freqz(taps, worN=500, fs=2)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(w, h.real, h.imag)
ax.set_xlabel('frequency')
ax.set_ylabel('Re(h)')
ax.set_zlabel('Im(h)')

plt.show()
# plt.savefig('plot_complex_freq_response_in_3d.svg')
