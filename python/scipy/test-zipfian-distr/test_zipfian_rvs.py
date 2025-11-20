"""
Test the Zipfian distribution random variates.

This module defines functions for testing the results
of scipy.stats.zipfian.rvs.
"""

from dataclasses import dataclass
import numpy as np
from scipy import stats
from scipy.stats import zipfian


def zipfian_aggregation_plan(a, n, p_min):
    pts = [1]  # First point in the support.
    while True:
        p1 = zipfian.cdf(pts[-1]-1, a, n) + p_min
        if p1 <= 1:
            k = int(zipfian.ppf(p1, a, n))
            pts.append(k+1)
        else:
            # Removing the last values results in the tail of the expected
            # array being merged into the previous computed bin.
            pts.pop()
            break
    # Return an array of indices ready to be used in np.add.reduceat().
    return np.array(pts) - 1


@dataclass
class Result:
    """Result class returned by run_power_divergence_tests()."""

    nbins: int
    min_expected_freq: float
    pvalues: np.ndarray


def run_one_test(rng, a, n, *, size, agg_indices, agg_expected):
    x = zipfian.rvs(a=a, n=n, size=size, random_state=rng)
    b = np.bincount(x, minlength=n + 1)[1:]
    b_agg = np.add.reduceat(b, agg_indices)
    test_result = stats.power_divergence(b_agg, agg_expected, lambda_=0)
    return test_result.pvalue


def run_nreps_tests(rng, a, n, size, agg_indices, agg_expected, nreps):
    pvalues = []
    for i in range(nreps):
        pvalues.append(run_one_test(rng, a, n, size=size, agg_indices=agg_indices, agg_expected=agg_expected))
    return pvalues


def run_power_divergence_tests(rng, a, n, *, size, nreps, min_freq=50,
                               show_progress=False):
    """
    bitgen: bit generator or generator from numpy.random
    a, n: Zipfian parameters
    size: number of samples per power divergence test
    nreps: number of repetitions of power divergence tests to run
        The length of the array of p-values returned will be nreps.
    min_freq: the minimum "frequency" to allow in the expected frequencies.
        This is used for binning the expected frequencies of the distribution.
    """
    p_min = min_freq/size
    if show_progress:
        print('Generating aggregation bins... ', end='', flush=True)
    indices = zipfian_aggregation_plan(a, n, p_min)
    if show_progress:
        print(f'nbins = {len(indices)}')
    k = np.arange(1, n + 1)  # Support of stats.zipfian.
    expected = size * zipfian.pmf(k, a, n)
    expected_agg = np.add.reduceat(expected, indices)

    if show_progress:
        print(f'Running {nreps} tests...')
    pvalues = []
    for i in range(nreps):
        pvalue = run_one_test(rng, a, n, size=size,
                              agg_indices=indices,
                              agg_expected=expected_agg)
        if show_progress:
            print(f'  {i = }  p = {pvalue}')
        pvalues.append(pvalue)
    return Result(pvalues=np.array(pvalues),
                  nbins=len(indices),
                  min_expected_freq=expected_agg.min())


def run_power_divergence_tests_multi(njobs, rng, a, n, *, size, nreps, min_freq=50):
    """
    njobs: number of parallel jobs to run
    rng: bit generator or generator from numpy.random
    a, n: Zipfian parameters
    size: number of samples per power divergence test
    nreps: number of repetitions of power divergence tests to run
        The length of the array of p-values returned will be nreps.
    min_freq: the minimum "frequency" to allow in the expected frequencies.
        This is used for binning the expected frequencies of the distribution.
    """
    from concurrent.futures import ProcessPoolExecutor

    p_min = min_freq/size
    indices = zipfian_aggregation_plan(a, n, p_min)
    k = np.arange(1, n + 1)  # Support of stats.zipfian.
    expected = size * zipfian.pmf(k, a, n)
    expected_agg = np.add.reduceat(expected, indices)

    rngs = rng.spawn(njobs)
    with ProcessPoolExecutor(max_workers=njobs) as executor:
        job_results = executor.map(run_nreps_tests,
                                   rngs,
                                   [a]*njobs,
                                   [n]*njobs,
                                   [size]*njobs,
                                   [indices]*njobs,
                                   [expected_agg]*njobs,
                                   [nreps]*njobs)

    pvalues = np.concat(list(job_results))
    return Result(pvalues=pvalues,
                  nbins=len(indices),
                  min_expected_freq=expected_agg.min())


def print_result(a, n, m, result, show_pvalues=True):
    print()
    print(f'{a = }  {n = }  {m = }')
    print(f'number of aggregation bins: {result.nbins}')
    print(f'minimum expected freq: {result.min_expected_freq:.2f}')
    if show_pvalues:
        print()
        print('p values')
        print('---------------------')
        for p in result.pvalues:
            print(p)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    __spec__ = None
    # rng = np.random.default_rng(121263137472525314065)
    rng = np.random.default_rng()

    # Zipfian parameters:
    a = 0.75
    n = 1250
    # Samples per test:
    m = 750000
    # Number of tests per parallel job:
    nreps = 25000
    # Number of parallel jobs:
    njobs = 4
    # Required minimum expected frequency for a bin when aggregating the
    # expected frequencies:
    min_freq = 100

    result = run_power_divergence_tests_multi(njobs, rng, a, n, size=m, nreps=nreps, min_freq=min_freq)
    print_result(a, n, m, result, show_pvalues=False)

    nbins = 50
    plt.hist(result.pvalues, bins=nbins)
    plt.grid(True)
    plt.xlabel('p')
    npvalues = len(result.pvalues)
    plt.title(f'Histogram with {nbins} bins of {npvalues} p-values from power divergence tests\n{a=} {n=} {m=}')
    plt.show()
