
import numpy as np
from scipy.special import gammaln


def loggamma1p(x):
    """
    Compute log(gamma(1 + x)) for real x.

    The implementation is designed to give accurate results for
    small x.  For example, for an array of very small numbers such
    as ``[-3e-16, -1e-18, 5e-30, 2.5e-20, 3e-16]``,
    ``scipy.special.gammaln(1 + x)`` loses all precision:

    >>> from scipy.special import gammaln
    >>> x = np.array([-3e-16, -1e-18, 5e-30, 2.5e-20, 3e-16])
    >>> gammaln(1 + x)
    array([ 4.44089210e-16,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
           -2.22044605e-16])

    (The precision is actually lost in the expression ``1 + x``.)

    ``loggamma1p`` computes these values correctly:

    >>> loggamma1p(x)
    array([ 1.73164699e-16,  5.77215665e-19, -2.88607832e-30, -1.44303916e-20,
           -1.73164699e-16])

    Note: Unlike ``scipy.special.gammaln``, the result of ``loggamma1p``
    always has type np.float64.  (This might be changed in the future.)

    """
    # In the interval -0.25 < x < 0.9, the relative error of
    # loggamma1p_small_x is generally less than 4e-16.

    # Some silly shenanigans so loggamma1p(<non-array scalar>) returns
    # np.float64, and not a scalar array.
    cast = False
    if not issubclass(type(x), np.ndarray):
        x = np.asarray(x)
        cast = True
    small_mask = (x > -0.25) & (x < 0.9)
    y = np.empty_like(x, dtype=np.float64)
    y[small_mask] = _loggamma1p_small_x(x[small_mask])
    y[~small_mask] = gammaln(1 + x[~small_mask])
    if cast and y.ndim == 0:
        y = np.float64(y)
    return y


# Coefficients of the Padé approximation to log(gamma(1+x)) at x=0.
_loggamma1p_small_x_p_coeff = [0.0, -0.5772156649015329, -2.198602239247181,
                               -2.8835804898328345, -0.7093852391116942,
                               2.054674619926225, 2.5151727627777807,
                               1.3458863118876616, 0.38837050891168406,
                               0.06011155167110235, 0.004451819276845639,
                               0.00011582239270882403, 2.362492383650223e-07]
_loggamma1p_small_x_q_coeff = [1.0, 5.23386570457449, 11.759169522860718,
                               14.820042213972009, 11.488581652651515,
                               5.650511133519242, 1.754785949617669,
                               0.33151383879069596, 0.0351480730651527,
                               0.0017788484304635968, 2.9167070790354156e-05]
# These two polynomials form the rational Padé approximation to
# log(gamma(1+x)) at x=0.
_loggamma1p_small_x_p = np.polynomial.Polynomial(_loggamma1p_small_x_p_coeff)
_loggamma1p_small_x_q = np.polynomial.Polynomial(_loggamma1p_small_x_q_coeff)


def _loggamma1p_small_x(x):
    return _loggamma1p_small_x_p(x) / _loggamma1p_small_x_q(x)
