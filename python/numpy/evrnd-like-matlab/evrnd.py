
import numpy as np


def evrnd(mu, sigma, size=None, rng=None):
    """
    Generate random variates from the Gumbel distribution.

    This function draws from the same distribution as the Matlab function

        evrnd(mu, sigma, n)

    `size` may be a tuple, e.g.

    >>> evrnd(mu=3.5, sigma=0.2, size=(2, 5))
    array([[3.1851337 , 3.68844487, 3.0418185 , 3.49705362, 3.57224276],
           [3.32677795, 3.45116032, 3.22391284, 3.25287589, 3.32041355]])

    """
    if rng is None:
        rng = np.random.default_rng()
    x = mu - rng.gumbel(loc=0, scale=sigma, size=size)
    return x
