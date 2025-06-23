
import numpy as np
import matplotlib.pyplot as plt


def foo(x, y):
    theta = 0.165
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    p = np.stack([x, y - 0.5])
    x2, y2 = np.einsum('ij,jkl->ikl', R, p)
    z = x2**2 + 450*y2**2
    return z


fig = plt.figure(figsize=(6, 4))
xx = np.linspace(-3.5, 3.5, 2500)
yy = np.linspace(-1.0, 2.0, 2500)
X, Y = np.meshgrid(xx, yy)
Z = foo(X, Y)
plt.contour(X, Y, Z, [2.5, 10.5], colors='k', linewidths=1)
plt.plot(0, 0.5, 'b.')
plt.plot([-3, 3], [0, 1], 'g.')
plt.grid()
plt.yticks([ -1, 0, 1, 2])
plt.axis('equal')

plt.show()
