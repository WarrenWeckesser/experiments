import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp


mp.dps = 100


def mp_var(c):
    c = mp.mpf(c)
    g1 = mp.gamma(1 - c)
    g2 = mp.gamma(1 - 2*c)
    return (g2 - g1*g1)/(c*c)

c, y1, y2 = np.loadtxt('out', unpack=True)

ref = []
for c1 in c:
    ref.append(float(mp_var(c1)))

ref = np.array(ref)
re1 = np.abs((y1 - ref)/ref)
re2 = np.abs((y2 - ref)/ref)

plt.plot(c, re1, label="g2 - g1**2")
plt.plot(c, re2, label="(sqrt(g2) - g1)*(sqrt(g2) + g1)")
plt.grid()
plt.legend(framealpha=1, shadow=True)

plt.show()
