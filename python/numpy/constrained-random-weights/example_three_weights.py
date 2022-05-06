
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from random_weights_with_ineq_constraints import (random_weights,
        compute_intersection_vertices, check_bounds)


def plot_box(ax, bounds):
    alpha = 0.5
    color = 'g'

    u = [0, 0, 1, 1]
    v = [0, 1, 1, 0]

    x0 = [bounds[0, 0]]*4
    y0 = [bounds[1, 0], bounds[1, 0], bounds[1, 1], bounds[1, 1]]
    z0 = [bounds[2, 0], bounds[2, 1], bounds[2, 1], bounds[2, 0]]
    tri = mtri.Triangulation(u, v)
    ax.plot_trisurf(x0, y0, z0, triangles=tri.triangles, linewidth=0.2,
                    antialiased=True, color=color, alpha=alpha)

    x0 = [bounds[0, 1]]*4
    y0 = [bounds[1, 0], bounds[1, 0], bounds[1, 1], bounds[1, 1]]
    z0 = [bounds[2, 0], bounds[2, 1], bounds[2, 1], bounds[2, 0]]
    tri = mtri.Triangulation(u, v)
    ax.plot_trisurf(x0, y0, z0, triangles=tri.triangles, linewidth=0.2,
                    antialiased=True, color=color, alpha=alpha)

    x0 = [bounds[0, 0], bounds[0, 0], bounds[0, 1], bounds[0, 1]]
    y0 = [bounds[1, 0]]*4
    z0 = [bounds[2, 0], bounds[2, 1], bounds[2, 1], bounds[2, 0]]
    tri = mtri.Triangulation(u, v)
    ax.plot_trisurf(x0, y0, z0, triangles=tri.triangles, linewidth=0.2,
                    antialiased=True, color=color, alpha=alpha)

    x0 = [bounds[0, 0], bounds[0, 0], bounds[0, 1], bounds[0, 1]]
    y0 = [bounds[1, 1]]*4
    z0 = [bounds[2, 0], bounds[2, 1], bounds[2, 1], bounds[2, 0]]
    tri = mtri.Triangulation(u, v)
    ax.plot_trisurf(x0, y0, z0, triangles=tri.triangles, linewidth=0.2,
                    antialiased=True, color=color, alpha=alpha)

    x0 = [bounds[0, 0], bounds[0, 0], bounds[0, 1], bounds[0, 1]]
    y0 = [bounds[1, 0], bounds[1, 1], bounds[1, 1], bounds[1, 0]]
    z0 = [bounds[2, 0]]*4
    tri = mtri.Triangulation(u, v)
    ax.plot_trisurf(x0, y0, z0, triangles=tri.triangles, linewidth=0.2,
                    antialiased=True, color=color, alpha=alpha)

    x0 = [bounds[0, 0], bounds[0, 0], bounds[0, 1], bounds[0, 1]]
    y0 = [bounds[1, 0], bounds[1, 1], bounds[1, 1], bounds[1, 0]]
    z0 = [bounds[2, 1]]*4
    tri = mtri.Triangulation(u, v)
    ax.plot_trisurf(x0, y0, z0, triangles=tri.triangles, linewidth=0.2,
                    antialiased=True, color=color, alpha=alpha)


bounds = np.array([[0.05, 0.2],
                   [0.25, 0.65],
                   [0.4, 0.6]])

check_bounds(bounds)

constrained_points = compute_intersection_vertices(bounds)
cmin = constrained_points.min(axis=0)
cmax = constrained_points.max(axis=0)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

ax = plt.figure().add_subplot(projection='3d')

ax.plot_trisurf(*np.eye(3), linewidth=0.2, antialiased=True, alpha=0.125)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plot_box(ax, bounds)

v = compute_intersection_vertices(bounds)
# This plot relies on the points in v being returned in an
# order that nicely encloses the intersection region.
# The code in compute_intersection_vertices() was not intentionally
# designed to do that, but it happens to work that way.
ax.plot(v[:, 0], v[:, 1], v[:, 2], c='g')
ax.plot(v[[-1, 0], 0], v[[-1, 0], 1], v[[-1, 0], 2], c='g')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


rng = np.random.default_rng(2342389820446)
nsamples = 500
samples = random_weights(bounds, nsamples, rng)

x1, y1, z1 = samples.T

ax.scatter(x1, y1, z1, s=1, c='k')

ax.set_xlim3d(0, 1)
ax.set_ylim3d(0, 1)
ax.set_zlim3d(0, 1)

plt.show()
