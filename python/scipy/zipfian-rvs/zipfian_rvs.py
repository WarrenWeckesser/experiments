import numpy as np
from scipy.stats import zipfian


# The approach used in zipfian_rvs() could be used in any discrete
# distribution with finite support.  But as noted in the docstring,
# it will be slow when the size of the support is large.

def zipfian_rvs(a, n, size=None, rng=None):
    """
    Generate random variates from the "Zipfian" distribution.

    The parameters `a` and `n` are the same as those of `scipy.stats.zipfian`.

    This function generates the length `n` array of probabilities of the
    distribution, and passes that as the `p` parameter to the `choice()`
    method of `rng`.  This means there is a lot of overhead in this
    function.  It is most effective for small to moderate values of `n`.
    The overhead cost is amortized when large values of `size` are used.
    """
    if rng is None:
        rng = np.random.default_rng()
    k = np.arange(1, n + 1)
    p = zipfian.pmf(k, a, n)
    return rng.choice(k, p=p, size=size)
