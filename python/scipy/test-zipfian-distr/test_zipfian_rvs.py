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


def run_power_divergence_tests(rng, a, n, *, size, nreps, min_freq=50,
                               show_progress=False):
    """
    bitgen: bit generator from numpy.random
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
        if show_progress:
            print(f'{i+1:3}  generating sample... ', end='', flush=True)
        x = zipfian.rvs(a=a, n=n, size=m, random_state=rng)
        b = np.bincount(x, minlength=n + 1)[1:]
        b_agg = np.add.reduceat(b, indices)
        test_result = stats.power_divergence(b_agg, expected_agg, lambda_=0)
        if show_progress:
            print(f'  p = {test_result.pvalue}')
        pvalues.append(test_result.pvalue)
    return Result(pvalues=np.array(pvalues),
                  nbins=len(indices),
                  min_expected_freq=expected_agg.min())


if __name__ == "__main__":
    # rng = np.random.default_rng(121263137472525314065)
    rng = np.random.default_rng()
    a = 1.25
    n = 500
    m = 500000
    nreps = 15
    min_freq = 100
    show_progress = False
    show_pvalues = True

    result = run_power_divergence_tests(rng, a, n, size=m, nreps=nreps,
                                        min_freq=min_freq,
                                        show_progress=show_progress)

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
    else:
        print(f"{nreps} computed p-values are in `result.pvalues`.")
