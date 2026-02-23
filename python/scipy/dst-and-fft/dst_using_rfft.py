#
# Compute the discrete sine transforms (types 1, 2, 3 and  4) with real
# input using rfft.
# See https://en.wikipedia.org/wiki/Discrete_sine_transform for how to
# interpret the types.
#

import numpy as np
from numpy.testing import assert_allclose
from scipy.fft import rfft, dst


# x is the input array.
x = np.array([1.5, 1, 2, 3, 5, 4, 1.24])

# Type 1
xx = np.concatenate(([0], x, [0], -x[::-1]))
y1 = -rfft(xx).imag[1:-1]
y2 = dst(x, type=1)
assert_allclose(y1, y2, rtol=1e-13)

# Type 2
xx = np.concatenate((x, -x[::-1]))
xz = np.zeros(2*len(xx))
xz[1::2] = xx
y1 = -rfft(xz).imag[1:len(x) + 1]
y2 = dst(x, type=2)
assert_allclose(y1, y2, rtol=1e-13)

# Type 3
xe = np.concatenate((x, x[-2::-1]))
xx = np.concatenate(([0], xe, [0], -xe[::-1]))
y1 = -rfft(xx).imag[1:-1:2] / 2
y2 = dst(x, type=3)
assert_allclose(y1, y2, rtol=1e-13)

# Type 4
xe = np.concatenate((x, x[::-1]))
xx = np.concatenate((xe, -xe))
xz = np.zeros(2*len(xx))
xz[1::2] = xx
y1 = -rfft(xz).imag[1:len(xe):2] / 2
y2 = dst(x, type=4)
assert_allclose(y1, y2, rtol=1e-13)
