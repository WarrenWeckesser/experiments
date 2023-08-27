from mpmath import mp


def _pdf_integrand(t, x):
    """
    Integrand of the PDF of as given in the wikipedia article,
    with mu = 0 and c = pi/2.
    """
    return mp.exp(-t)*mp.cos(2*t*x/mp.pi + 2*t/mp.pi*mp.log(2*t/mp.pi))


def landau_pdf(x):
    """
    "Standard" Landau distribution, corresponding to mu = 0 and c = pi/2
    in the formula given in the wikipedia article
         https://en.wikipedia.org/wiki/Landau_distribution
    """
    x = mp.mpf(x)
    return (2/mp.pi**2)*mp.quad(lambda t: _pdf_integrand(t, x), [0, mp.inf])


def _pdf_deriv_integrand(t, x, c):
    return mp.exp(-t)*mp.sin(t*x/c + 2*t/mp.pi*mp.log(t/c))*t/c


def landau_pdf_deriv(x):
    x = mp.mpf(x)
    c = mp.pi/2
    return -mp.quad(lambda t: _pdf_deriv_integrand(t, x, c),
                    [0, mp.inf])/(mp.pi*c)


mp.dps = 250
landau_xstar = mp.findroot(landau_pdf_deriv, -0.22278)
print(f"landau_xstar = {float(landau_xstar)}")
