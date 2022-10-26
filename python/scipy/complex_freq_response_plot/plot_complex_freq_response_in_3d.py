import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, remez


taps = remez(31, [0, 0.3, 0.4, 1], [1, 0], fs=2)

w, h = freqz(taps, worN=500, fs=2)

ax1 = plt.subplot()
ax1.plot(w, np.abs(h))
ax1.set_xlabel('frequency')
ax1.set_ylabel('gain, abs(h)')
ax1.grid(True)

plt.savefig('plot_complex_freq_response_in_3d_fig1.svg')

fig = plt.figure()
ax2 = fig.add_subplot(projection='3d')
ax2.plot(w, h.real, h.imag)
ax2.set_xlabel('frequency')
ax2.set_ylabel('Re(h)')
ax2.set_zlabel('Im(h)')

plt.savefig('plot_complex_freq_response_in_3d_fig2.svg')

plt.show()
