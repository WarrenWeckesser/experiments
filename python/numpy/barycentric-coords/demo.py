import numpy as np
from barycentric_coords import barycentric_coords


def demo(pts, simplex):
    print("simplex:")
    print(simplex)
    print("points:")
    print(pts)
    w = barycentric_coords(pts, simplex)
    print("barycentric coordinates:")
    print(w)
    print("recreate points from the coordinates:")
    print(w @ simplex)


simplex = np.array([[1.0, 0.0], [4.0, 2.0], [0.0, 3.0]])

pt = np.array([1.5, 1.5])
demo(pt, simplex)

print('-----')

pts = np.array([[0.25, 0.125], [1, 2], [-1, 2], [1.25, 2.0]])
demo(pts, simplex)

print('-----')

simplex = np.array([[1, 1, 1], [3, 5, 2], [-3, -1, 3], [5, -4, 0.5]])
pts = np.array([[1, 2, 3],
                [1, 1, 1],
                [4, 3, 2],
                0.75*simplex[0] + 0.25*simplex[2],
                np.mean(simplex, axis=0)])
demo(pts, simplex)
