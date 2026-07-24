from mpmath import mp
import numpy as np
from scipy.fft import dct
import matplotlib.pyplot as plt
from chebstuff import chebyshev_coefficients, chebyshev_eval, chebyshev_lobatto_sample


def foo(x):
    return np.exp(-x) * np.sin(x)


def foo_mp(x):
    return mp.exp(-x) * mp.sin(x)


# For the problem as currently implemented, increasing N beyond 17 doesn't
# decrease the max_relerr computed below.
N = 17
a = 1
b = 3

mp.dps = 250

t, x, y = chebyshev_lobatto_sample(mp, foo_mp, a, b, N)
mp_coefs = chebyshev_coefficients(mp, y, N)
coefs = [float(q) for q in mp_coefs]

xx = np.linspace(a, b, 500)
yy = foo(xx)

tt = (2.0*xx - (a + b)) / (b - a)
print(f"coefs: {coefs}")
yyi = [chebyshev_eval(t1, coefs) for t1 in tt]
yy_ref = [foo_mp(x1) for x1 in xx]

# Max relative error in the interpolated values...
max_relerr = float(max(abs((np.array(yyi) - yy_ref)/yy_ref)))
print(f"{max_relerr = }")

plt.plot(xx, yy, label="Original function")
plt.plot(xx, yyi, linewidth=5, alpha=0.5, label="Chebyshev polynomial approximation")
plt.legend(framealpha=1, shadow=True)
plt.grid()
plt.show()
