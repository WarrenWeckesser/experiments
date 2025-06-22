
import numpy as np
import matplotlib.pyplot as plt


def slerp1(p0, p1, t):
    # This function assumes p0, p1 and t are 1-d.

    p0 = np.asarray(p0)
    p1 = np.asarray(p1)
    t = np.asarray(t)

    omega = np.arccos(np.dot(p0, p1))
    t = t.reshape(-1, 1)
    z = (np.sin((1 - t)*omega)*p0 + np.sin(t*omega)*p1) / np.sin(omega)

    return z


def slerp2(p0, p1, t):
    # This version assumes p0 and p1 are 1-d, but
    # t can be n-d.  The return value has shape
    #    t.shape + (len(p0),).

    p0 = np.asarray(p0)
    p1 = np.asarray(p1)
    t = np.asarray(t)

    omega = np.arccos(np.dot(p0, p1))
    t = np.asarray(t)
    shp = t.shape
    t = t.reshape(-1, 1)
    z = (np.sin((1 - t)*omega)*p0 + np.sin(t*omega)*p1) / np.sin(omega)

    return z.reshape(shp + (len(p0),))


def slerp3(p0, p1, t):
    # This version assumes p0 and p1 are 1-d, but
    # t can be n-d.  The return value has shape
    #    t.shape + (len(p0),).

    p0 = np.asarray(p0)
    p1 = np.asarray(p1)
    t = np.asarray(t)

    omega = 2*np.pi - np.arccos(np.dot(p0, p1))
    t = np.asarray(t)
    shp = t.shape
    t = t.reshape(-1, 1)
    z = (np.sin((1 - t)*omega)*p0 + np.sin(t*omega)*p1) / np.sin(omega)

    return z.reshape(shp + (len(p0),))


p0 = np.array([1, 0])
p1 = np.array([0, 1])
t = np.array([0, 0.1, 0.25, 0.75, 1])
s2 = slerp2(p0, p1, t)
s3 = slerp3(p0, p1, t)

theta = np.linspace(0, 2*np.pi, 400)
plt.plot(np.cos(theta), np.sin(theta), 'k', alpha=0.25)
plt.plot(s2[:,0], s2[:, 1], 'k.')
plt.plot(s3[:,0], s3[:, 1], 'ro', alpha=0.3, markersize=8)
plt.axis('equal')
plt.show()
