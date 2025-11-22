
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import boxcox
# Private function; never rely on this in "production" code...
from scipy.special._ufuncs import _gen_harmonic


def m_const(a, n):
    return (boxcox(n, 1 - a) + 1) / _gen_harmonic(n, a)


ls = ['-', '--', ':']
a = np.linspace(0, 10, 1500)
n = [2, 3, 10, 100, 1000, 10000]
for k, n1 in enumerate(n):
    plt.plot(a, m_const(a, n1), label=f'n = {n1}', linestyle=ls[k % len(ls)])
plt.xlabel('a')
plt.legend(framealpha=1, shadow=True)
plt.title('Rejection method dominating PDF scaling constant $M(a, n)$')
plt.grid()
plt.show()
