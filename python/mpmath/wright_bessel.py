# Links:
#   https://dlmf.nist.gov/10.46
#   https://en.wikipedia.org/wiki/Bessel%E2%80%93Maitland_function

from mpmath import mp


def _wright_bessel_term(k, z, rho, beta):
    numer = mp.power(z, k)
    denom = mp.gamma(k*rho + beta)*mp.factorial(k)
    return numer / denom


def wright_bessel(z, rho, beta):
    """
    Wright's generalized Bessel function.

    Also known as the Bessel-Maitland function.

    Parameter convention corresponds to

        https://dlmf.nist.gov/10.46#E1

    but here the `z` parameter is given first.
    """
    if z == 0:
        return 1 / mp.gamma(beta)
    return mp.nsum(lambda k: _wright_bessel_term(k, z, rho, beta),
                   [0, mp.inf])


def wright_bessel_rho1(x, beta):
    """
    Wright's generalized Bessel function.

    Also known as the Bessel-Maitland function.

    Parameter convention corresponds to

        https://dlmf.nist.gov/10.46#E1

    but here the `x` parameter is given first, and rho is fixed at 1.
    `x` is assumed to be real.
    """
    beta = mp.mpf(beta)
    nu = beta - 1
    if x > 0:
        r = mp.sqrt(x)
        w = mp.besseli(nu, 2*r) / mp.power(r, nu)
    elif x < 0:
        r = mp.sqrt(-x)
        w = mp.besselj(nu, 2*r) / mp.power(r, nu)
    else:
        # x == 0
        w = 1 / mp.gamma(beta)
    return w


def _wright_bessel_term_wp(k, z, mu, nu):
    # The '_wp' suffix stands for "wikipedia".  The parameter convention
    # corresponds to the parameters given in the wikipedia article.
    numer = mp.power(-z, k)
    denom = mp.gamma(k*mu + nu + 1)*mp.factorial(k)
    return numer / denom


def wright_bessel_wp(z, mu, nu):
    """
    Wright's generalized Bessel function.

    Also known as the Bessel-Maitland function.

    The parameters correspond to the formula from the wikipedia
    aricle

        https://en.wikipedia.org/wiki/Bessel%E2%80%93Maitland_function

    """
    return mp.nsum(lambda k: _wright_bessel_term_wp(k, z, mu, nu),
                   [0, mp.inf])
