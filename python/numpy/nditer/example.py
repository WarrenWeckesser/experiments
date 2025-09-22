# Scratch work -- experimenting with nditer.

import numpy as np


def foo(x, p):
    k = 0
    for x1, p1 in np.nditer([x, p], flags=['external_loop']):
        print(f'{k = }  {x1 = }  {p1 = }')
        k += 1


def minmax(x):
    x = np.asarray(x)
    dt = np.dtype([('min', x.dtype), ('max', x.dtype)])
    out = np.zeros(1, dtype=dt)
    it = np.nditer([x, out], flags=['reduce_ok', 'external_loop'],
                             op_flags=[['readonly', 'no_broadcast'], ['readwrite', 'no_broadcast']])
    with it:
        for xi, outi in it:
            print(f'{xi = }    {outi = }')
            #outi[0] = xi.min()
            #outi[1] = xi.max()
    return out


def trythis(a, b):
    with np.nditer([a, b], op_axes=[[0], [0]]) as it:
        for x, y in it:
            print(f"{x = }   {y = }")



# x = np.array([[1, 2, 3], [8, 9, 10]])
# p = np.array([0.25, 0.50, 0.25])

x = np.array([[1], [2]])
p = np.arange(1000000).reshape((1, -3))
foo(x, p)
