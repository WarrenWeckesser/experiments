import numpy as np
from scipy.stats import rankdata
from scipy.special import ndtri, chdtrc


def fk_chi2(*samples, center='median'):
    """
    Conover et al's interpretation of Fligner and Killeen's analog of Klotz.

    This function is similar to `scipy.stats.fligner`.

    With the default `center="median"`, it computes the statistic annotated
    in Conover et al [1] as 'F-K:med X²', which is inspired by--but not quite
    the same as--the third statistic described by Fligner and Killeen [2].
    Fligner and Killeen described that statistic as an analog of the statistic
    defined by Klotz [3].
    
    This calculation (with the default `center="median"`) is the same as
    that of the R function `fligner.test` from the R `stats` package.

    Conover et al's change is to apply the centering function to each sample
    independently.  In the statistics described by Fligner and Killen, the
    center value is computed as a single value based on the combined samples.

    References
    ----------
    [1] W. J. Conover, M. E. Johnson, and M. M. Johnson, "A Comparative Study
        of Tests for Homogeneity of Variances, with Applications to the Outer
        Continental Shelf Bidding Data", Technometrics, Vol. 23, No. 4,
        November 1981, 351-361.
    [2] M. A. Fligner and T. J. Killeen, "Distribution-Free Two-Sample Tests
        for Scale", Journal of the American Statistical Association, March 1976,
        Volume 71, Number 353, 210-213.
    [3] J. Klotz, "Nonparametric Tests for Scale", The Annals of Mathematical
        Statistics, 32 (June 1962), 498-512.
    """
    if center not in ['median', 'mean']:
        if not callable(center):
            raise ValueError("center must be 'median', 'mean' or a callable "
                             "object with signature f(x) that computes a central "
                             "tendency statistic for the values in the 1-d "
                             f"array x; got {repr(center)}")
    
    if center == 'median':
        func = np.median
    elif center == 'mean':
        func = np.mean
    else:
        func = center

    num_samples = len(samples)
    samples = [np.asarray(sample) for sample in samples]
    sizes = np.array([sample.size for sample in samples])
    abs_centered_samples = [np.abs(sample - func(sample)) for sample in samples]
    u = np.concat(abs_centered_samples)
    # Klotz transform [3];  Conover et al [1], Table 4, last entry in the second column;
    # Killeen and Fligner [1], in formula for T_{N, 3}.
    N = len(u)
    r = rankdata(u)
    a = ndtri(0.5 + r/(2*(N + 1)))

    # Compute the χ² statistc (Conover et al [1], equation (2.1)).
    abar = np.mean(a)
    rngs = np.concat(([0], np.cumsum(sizes[:-1])))
    mean_group_scores = np.add.reduceat(a, rngs)/sizes
    v2 = (1/(N - 1))*np.sum((a - abar)**2)
    chi2 = (sizes @ (mean_group_scores - abar)**2)/v2
    df = num_samples - 1
    pvalue = chdtrc(df, chi2)
    return chi2, df, pvalue
