import numpy as np
from scipy import stats
from random_variates import zipfian


def zipfian_aggregation_plan(a, n, p_min):
    pts = [1]  # First point in the support.
    while True:
        p1 = stats.zipfian.cdf(pts[-1]-1, a, n) + p_min
        if p1 <= 1:
            k = int(stats.zipfian.ppf(p1, a, n))
            pts.append(k+1)
        else:
            tmp = pts.pop()
            break
    # Return an array of indices ready to be used in np.add.reduceat().
    return np.array(pts) - 1


if __name__ == "__main__":
    bitgen = np.random.PCG64()

    a = 1.2
    n = 100000
    m = 10000000

    p_min = 50/m
    indices = zipfian_aggregation_plan(a, n, p_min)
    k = np.arange(1, n + 1)  # Support of stats.zipfian.
    expected = m * stats.zipfian.pmf(k, a, n)
    expected_agg = np.add.reduceat(expected, indices)

    print(f'{a = }  {n = }  {m = }')
    print(f'number of aggregation bins: {len(indices)}')
    print(f'minimum expected freq: {expected_agg.min():.2f}')
    print()
    print('p values')
    print('---------------------')
    nreps = 20
    for i in range(nreps):
        x = zipfian(bitgen, a=a, n=n, size=m)
        b = np.bincount(x, minlength=n + 1)[1:]
        b_agg = np.add.reduceat(b, indices)
        test_result = stats.power_divergence(b_agg, expected_agg, lambda_=0)
        print(test_result.pvalue)
