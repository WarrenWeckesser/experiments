import sys
import numpy as np
import matplotlib.pyplot as plt


def dconverter(s):
    return float(s.replace('D', 'e'))


if len(sys.argv) < 2:
    print("The data filename to be plotted is required.", file=sys.stdout)
    exit(-1)
elif len(sys.argv) > 2:
    print("Only one command line argument, the data filename, is accepted.",
          file=sys.stderr)
    exit(-1)

converters = {k: dconverter for k in range(4)}
t, S, I, R = np.loadtxt(sys.argv[1], converters=converters, unpack=True,
                        encoding='ASCII')

plt.plot(t, S, label='S')
plt.plot(t, I, '--', label='I')
plt.plot(t, R, dashes=(6, 2, 3, 2), label='R')
plt.legend(framealpha=1, shadow=True)
plt.grid()
plt.xlabel('t')
plt.show()
