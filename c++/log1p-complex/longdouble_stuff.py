# WIP/scratch work...

# Trying to emulate the double-longdouble C code.
# This assumes that "long double" is 80 bit ("extended precision").

import numpy as np


def longdouble_split(x):
    t = np.longdouble(((1 << 32) + 1))*x
    upper = t - (t - x)
    lower = x - upper
    return upper, lower


def longdouble_square(x):
    split_upper, split_lower = longdouble_split(x)
    out_upper = x*x
    # out_lower = (split_upper*split_upper - out_upper
    #               + np.longdouble(2)*split_upper*split_lower
    #               + split_lower*split_lower)

    out_lower = (split_lower*split_lower
                 - ((out_upper - split_upper*split_upper)
                   - 2*split_lower*split_upper))

    return out_upper, out_lower


def longdouble_two_sum_quick(x, y):
    r = x + y
    e = y - (r - x)
    return r, e


def longdouble_two_sum(x, y):
    s = x + y
    v = s - x
    e = (x - (s - v)) + (y - v)
    return s, e


def longdouble_double_sum(xd, yd):
    out_upper, out_lower = longdouble_two_sum(xd[0], yd[0])
    out_lower += xd[1] + yd[1]
    out_upper, out_lower = longdouble_two_sum_quick(out_upper, out_lower)
    return out_upper, out_lower


def longdouble_xsquared_plus_2x_plus_ysquared(x, y):
    x2 = longdouble_square(x)
    y2 = longdouble_square(y)
    twox_upper = 2*x
    twox_lower = type(x)(0)
    sum1 = longdouble_double_sum(x2, (twox_upper, twox_lower))
    sum2 = longdouble_double_sum(sum1, y2)
    return sum2[0]
