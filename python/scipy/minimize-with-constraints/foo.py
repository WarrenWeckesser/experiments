import numpy as np


def foo(x):
    return np.sum((x - 1)**2)


def foo_strict_bounds(x, b):
    res0, res1 = b.residual(x)
    if np.any(res0 < 0) or np.any(res1 < 0):
        raise RuntimeError("constraint violated")
    return np.sum((x - 1)**2)


def foojac(x):
    return 2*(x - 1)


def foohess(x):
    return 2*np.eye(len(x))
