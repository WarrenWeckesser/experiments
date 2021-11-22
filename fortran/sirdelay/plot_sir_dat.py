import numpy as np
import matplotlib.pyplot as plt


def dconverter(s):
    return float(s.replace('D', 'e'))


converters = {k: dconverter for k in range(4)}

t, S, I, R = np.loadtxt('sir2.dat', converters=converters, unpack=True,
                        encoding='ASCII')

plt.plot(t, S, label='S')
plt.plot(t, I, label='I')
plt.plot(t, R, label='R')
plt.legend(framealpha=1, shadow=True)
plt.grid()
plt.show()
