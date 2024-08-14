
import numpy as np


# tl_invcdf is for relatively small lambda.
# I don't have a precise definition for "relatively small".
# p == 0 and p == 1 are not handled.

def tl_invcdf(p, lam, n=None):
    if p == 0.5:
        return np.float64(0.0), 0
    # This method computes a product of n terms. The initial value
    # of the product is log(p/(1 - p)) (equivalently log(p) - log1p(-p)).
    # When p is close to 0.5, there is a nontrivial amount of precision
    # lost in the expression log(p) - log1p(-p), so instead the degree 7
    # Taylor polynomial about p=0.5 is used.
    if abs(p - 0.5) < 0.01:
        ps = p - 0.5
        ps2 = ps**2
        x = np.float64(ps*(4 +
                           ps2*(5.333333333333333 +
                                ps2*(12.8 +
                                     36.57142857142857*ps2))))
    else:
        x = np.log(p) - np.log1p(-p)
    x = np.log(p) - np.log1p(-p)
    if n is not None:
        for k in range(1, n + 1):
            b = lam/2**k
            x *= (p**b + (1 - p)**b)/2
        return x, n
    k = 1
    while k < 50:
        b = lam/2**k
        f = (p**b + (1 - p)**b)/2
        # print(f'{k:3} {x}  {f}')
        nextx = x*f
        if abs((x - nextx)/nextx) < 5e-15:
            return nextx, k
        x = nextx
        k += 1
    return x, k
