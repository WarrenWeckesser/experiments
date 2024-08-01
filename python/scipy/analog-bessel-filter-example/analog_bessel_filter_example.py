import numpy as np
from scipy.signal import bessel, lsim
import matplotlib.pyplot as plt


# Create a low-pass filter, with a cutoff of 16 Hz.
b, a = bessel(N=6, Wn=2*np.pi*16, btype='lowpass', output='ba', analog=True)

# Generate data to which the filter is applied.
t = np.linspace(0, 1.25, 500, endpoint=False)
# Sum of three sinusoidal curves, with frequencies 4 Hz, 40 Hz, and 80 Hz.
# The filter should mostly eliminate the 40 Hz and 80 Hz components,
# leaving just the 4 Hz signal.
u = np.cos(2*np.pi*4*t) + np.sin(2*np.pi*40*t) + 0.5*np.cos(2*np.pi*80*t)

tout, yout, xout = lsim((b, a), U=u, T=t)

plt.plot(t, u, 'r', alpha=0.3, linewidth=1, label='input')
plt.plot(t, yout, 'b', alpha=0.4, linewidth=4, label='lsim output')
plt.legend(loc='best', shadow=True, framealpha=1)
plt.grid(alpha=0.3)
plt.show()
