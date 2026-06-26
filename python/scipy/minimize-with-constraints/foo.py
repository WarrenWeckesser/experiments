import numpy as np


def foo(x):
    return np.sum((x - 1)**2)


def foojac(x):
    return 2*(x - 1)


def foohess(x):
    return 2*np.eye(len(x))
