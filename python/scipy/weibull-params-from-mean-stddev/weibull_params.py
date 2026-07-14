
import numpy as np
from scipy.special import gamma, gammaln
from scipy.optimize import fsolve, root
from scipy.stats import weibull_min


def _h(c):
    r = np.exp(gammaln(2/c) - 2*gammaln(1/c))
    return np.sqrt(1/(2*c*r - 1))


def weibull_c_scale_from_mean_std(mean, std):
    """
    Given the desired mean and std. dev. for the Weibull (min.) distribution,
    compute the corresponding shape parameter c and the scale parameter.

    This is for the two-parameter Weibull distribution--the location parameter
    is assumed to be 0.

    The solvers used in this function may fail for certain parameter ranges.
    A RuntimeError is raised when that occurs.

    Implementation details:
    The function makes two attempts to solve for the parameters.  The
    first attempt uses `scipy.optimize.fsolve`.  If that fails, it tries
    `scipy.optimize.root` with `method='lm'`.
    """
    c0 = 1.27*np.sqrt(mean/std)
    c, info, ier, msg = fsolve(lambda t: _h(t) - (mean/std), c0, xtol=1e-10,
                               full_output=True)
    if ier != 1:
        result = root(lambda t: _h(t) - (mean/std), c0, method='lm')
        if not result.success:
            errmsg = ("fsolve() and root() failed to solve for the parameters.\n"
                      f"fsolve: {msg = }\nroot: {result.message = }")
            raise RuntimeError(errmsg)
        c = result.x
    c = c[0]
    scale = mean / gamma(1 + 1/c)
    return c, scale


def weibull_c_and_scale_test(mean=(1,)):
    """
    This function validates `weibull_c_scale_from_mean_std` for each value of
    `mean`, testing the behavior for ten values of standard deviation spanning
    0.1 to 100.
    """
    for mn in mean:
        print("       mn         mn_actual        sd        sd_actual"
              "           c             scale")
        format = 4 * '{:13.9f} ' + 2 * '{:18.12g}'

        for sd in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]:
            c, scale = weibull_c_scale_from_mean_std(mn, sd)
            mn_actual = weibull_min(c=c, scale=scale).mean()
            sd_actual = weibull_min(c=c, scale=scale).std()
            print(format.format(mn, mn_actual, sd, sd_actual, c, scale))
