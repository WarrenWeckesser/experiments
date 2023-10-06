
import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp


mp.dps = 250

# This assumes that 'out' was created by running hyp1f1_sample and
# directing the output to the file 'out', e.g.
# $ ./hyp1f1_sample 13 1.48 1.52 5000 55 > out
data = np.loadtxt('out')

err1 = []
err2 = []
for a, b, x, h1, h2 in data:
    h = mp.hyp1f1(a, b, x)
    err1.append(float(abs(h - h1)/h))
    err2.append(float(abs(h - h2)/h))


b = data[:,1]

plt.plot(b, err1, alpha=0.65, label='$_{1}F_{1}$ rel. error')
plt.plot(b, err2, alpha=0.65, label='$_{p}F_{q}$ rel. error')
plt.xlabel('b')
plt.title(f'Relative error of ₁F₁({a}, b, {x})')
plt.grid()
plt.legend(shadow=True, framealpha=1)
plt.semilogy()
plt.show()
