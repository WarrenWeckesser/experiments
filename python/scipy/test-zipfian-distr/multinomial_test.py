import numpy as np
import operator
from scipy.special import logsumexp
from scipy.stats import multinomial, zipfian


# This is slower than list(multinomial_support_generator(n, m)).
def multinomial_support(n, m):
    if n == 0:
        return [(0,)*m]
    if m == 1:
        return [(n,)]
    supp = []
    for k in range(n, -1, -1):
        supp1 = multinomial_support(n - k, m - 1)
        supp.extend([(k,) + s for s in supp1])
    return supp


def multinomial_support_generator(n, m):
    n = operator.index(n)
    m = operator.index(m)
    if m < 1:
        raise ValueError('m must be a positive integer.')
    if n < 0:
        raise ValueError('n must be nonnegative')
    b = [0]*m
    b[0] = n
    yield b[:]
    while True:
        k = b[-1]
        if k == n:
            return
        b[-1] = 0
        i = -2
        while b[i] == 0:
            i -= 1
        b[i] -= 1
        b[i+1] = 1 + k
        yield b[:]


def all_possible_pvalues(n, a, m):
    """
    Tractable for very small n and m only!
    For example, the following shows the number of points in the support
    for various values of n and m.

        n     m    len(support)
        5     10        2002
        5     25      118755
        5     40     1086008
        8     15      170544
        9     15      490314
       10     15     1307504
    """
    zpmf = zipfian.pmf(np.arange(1, n + 1), a, n)
    mn = multinomial(m, zpmf)
    supp = list(multinomial_support_generator(m, n))
    mn_logpmf = mn.logpmf(supp)
    logpvalues = []
    for v in supp:
        logp0 = mn.logpmf(v)
        mask = mn_logpmf <= logp0
        logp = logsumexp(mn_logpmf[mask])
        logpvalues.append(logp)
    return supp, np.array(logpvalues)

#
# multinomial_test() is not used in run_zipfian_tests().
# The test is inlined there to make it easy to reuse the support array.
#
def multinomial_test(k, p):
    if len(k) != len(p):
        raise ValueError('len(k) != len(p)')
    m = np.sum(k)
    mn = multinomial(m, p)
    supp = np.array(list(multinomial_support_generator(m, len(p))))
    logp0 = mn.logpmf(k)
    logpmf = mn.logpmf(supp)
    mask = np.exp(logpmf) <= np.exp(logp0)
    return logsumexp(logpmf[mask])


def multinomial_logpmf(a, n, m):
    zpmf = zipfian.pmf(np.arange(1, n + 1), a, n)
    mn = multinomial(m, zpmf)
    supp = np.array(multinomial_support(m, n))
    mn_logpmf = mn.logpmf(supp)
    return mn_logpmf


def run_zipfian_tests(a, n, m, ntests, verbose=False):
    """
    a and n are the parameters of the Zipfian distribution.
    m is the number of variates to generate for each test.
    ntests is the number of tests to run.

    Returns the log of the p-value for each multinomial test.
    """
    zpmf = zipfian.pmf(np.arange(1, n + 1), a, n)
    mn = multinomial(m, zpmf)

    if verbose:
        print("Generating multinomial support array")
    supp = np.array(list(multinomial_support_generator(m, n)))
    mn_logpmf = mn.logpmf(supp)
    if verbose:
        print(f"Support array has length {len(mn_logpmf)}")

    pvalues = []
    if verbose:
        print(f"Running {ntests} tests")
    for i in range(ntests):
        if verbose:
            print(f"{i = :5}  ", end='')
        sample = zipfian.rvs(a, n, size=m)
        b = np.bincount(sample, minlength=n + 1)[1:]
        if verbose:
            print(f"{b = !s}  ", end='')
        # Multinomial test
        logp0 = mn.logpmf(b)
        mask = mn_logpmf <= logp0
        logp = logsumexp(mn_logpmf[mask])
        if verbose:
            print(f"log(p) = {logp!s}")
        pvalues.append(logp)

    return np.array(pvalues)
