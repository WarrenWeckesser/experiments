"""Utilities for engineering notation."""

from math import log10, floor


def eng(x):
    """
    Express a number in engineering notation.

    Returns the tuple (y,p) such that x = y*10**p,
    1 <= abs(y) < 1000, and p is a multiple of 3.
    """

    if x == 0.0:
        return 0.0, 0
    sign = 1
    if x < 0.0:
        sign = -1
        x = -x
    p = int(floor(log10(x) / 3.0) * 3)
    y = x / (10.0**p)
    return sign*y, p


def eng_format(x, width=7, decimals=3, ealways=False):
    """Create a string representation of x using engineering notation."""

    value, p = eng(x)
    if p != 0 or ealways:
        valstr = f'{value:{width}.{decimals}f}e{p}'
    else:
        valstr = f'{value:{width}.{decimals}f}'
    return valstr


if __name__ == "__main__":
    print(eng_format(1.2345e-1))
    print(eng_format(1.2345e0))
    print(eng_format(1.2345e0, ealways=True))
    print(eng_format(1.2345e1))
    print(eng_format(1.2345e1, ealways=True))
    print(eng_format(1.2345e2))
    print(eng_format(1.2345e3))
    print(eng_format(1.2345e4))
    print(eng_format(1.2345e5))
    print(eng_format(1.2345e6))
    print(eng_format(1.2345e7))
    print(eng_format(0.0))
    print(eng_format(-0.1))
    print(eng_format(-1.2345e10))
