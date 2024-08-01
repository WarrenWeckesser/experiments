# from https://stackoverflow.com/
#     questions/61250845/setting-wn-for-analog-bessel-model-in-scipy-signal

import numpy as np
from scipy.signal import bessel, step, impulse
import matplotlib.pyplot as plt


order = 4
Wn = 2*np.pi * 2000
b, a = bessel(order, Wn, btype='low', analog=True, output='ba', norm='phase')

# Note: the upper limit for t was chosen after some experimentation.
# If you don't give a T argument to impulse or step, it will choose a
# a "pretty good" time span.
t = np.linspace(0, 0.00125, 2500, endpoint=False)
timp, yimp = impulse((b, a), T=t)
tstep, ystep = step((b, a), T=t)


plt.subplot(2, 1, 1)
plt.plot(timp, yimp, label='impulse response')
plt.legend(loc='upper right', framealpha=1, shadow=True)
plt.grid(alpha=0.25)
plt.title('Impulse and step response of the Bessel filter')

plt.subplot(2, 1, 2)
plt.plot(tstep, ystep, label='step response')
plt.legend(loc='lower right', framealpha=1, shadow=True)
plt.grid(alpha=0.25)
plt.xlabel('t')
plt.show()
