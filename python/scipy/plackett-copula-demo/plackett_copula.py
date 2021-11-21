# Copyright 2021 Warren Weckesser

import numpy as np
from scipy.stats import weibull_min, spearmanr, pearsonr
from scipy.optimize import fsolve
from scipy.special import logit
import matplotlib.pyplot as plt


# Generate samples from the bivariate Plackett copula with Weibull
# marginal distributions.
#
# See Chapter 6 of the text *Copulas and their Applications in Water
# Resources Engineering*, Cambridge University Press (2019).


def spearman_rho_log(logtheta):
    """
    Compute the Spearman coefficient rho from the log of the
    theta parameter of the bivariate Plackett copula.
    """
    # The formula for rho from slide 66 of
    # http://www.columbia.edu/~rf2283/Conference/1Fundamentals%20(1)Seagers.pdf
    # rho is
    #   rho = (theta + 1)/(theta - 1) - 2*theta/(theta - 1)**2 * log(theta)
    # If R = log(theta), this can be rewritten as
    #   coth(R/2) - R/(cosh(R) - 1)
    #
    # Note, however, that the formula for the Spearman correlation rho in
    # the article "A compendium of copulas" at
    #   https://rivista-statistica.unibo.it/article/view/7202/7681
    # does not include the term log(theta).  (See Section 2.1 on the page
    # labeled 283, which is the 5th page of the PDF document.)

    b = 1/np.tanh(logtheta/2) - logtheta/(np.cosh(logtheta) - 1)
    return b


def est_logtheta(rho):
    # This function gives a pretty good estimate of log(theta) for
    # the given Spearman coefficient rho.  That is, it approximates
    # the inverse of spearman_rho_log(logtheta).
    return logit((rho + 1)/2)/0.69


def bivariate_plackett_theta(spearman_rho):
    """
    Given the Spearman rho coefficient (a value in [-1, 1]), compute
    the corresponding value of the parameter theta of the Plackett copula.
    """
    result = fsolve(lambda t: spearman_rho_log(t) - spearman_rho,
                    est_logtheta(spearman_rho), xtol=1e-10, full_output=True)
    logtheta1, fsolve_info, status, msg = result
    if status != 1:
        raise RuntimeError(f'failed to solve for logtheta: {msg}')
    return np.exp(logtheta1[0])


def bivariate_plackett_sample(theta, m, random=None):
    """
    Generate m samples from the bivariate Plackett copula with parameter theta.

    `random`, if given, must be an object with a `uniform` method that accepts
    a `size` keyword parameter and returns an array of sample from the standard
    uniform distribution with shape specified by `size`.

    Returns an array with shape (m, 2).
    """
    if random is None:
        rng = np.random.default_rng()
    else:
        rng = random

    # These calculations are based on the information in Chapter 6 of the text
    # *Copulas and their Applications in Water Resources Engineering*
    # (Cambridge University Press, 2019).
    u, w2 = rng.uniform(size=(2, m))
    # w2 = rng.uniform(size=m)
    S = w2*(1 - w2)
    d = np.sqrt(theta*(theta + 4*S*u*(1 - u)*(1 - theta)**2))
    c = 2*S*(u*theta**2 + 1 - u) + theta*(1 - 2*S)
    b = theta + S*(theta - 1)**2
    v = (c - (1 - 2*w2)*d)/(2*b)
    return np.column_stack((u, v))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Main calculation
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Desired Spearman coefficient
rho = -0.75
# Number of samples to generate.
m = 5000

# Get the value of theta of the Plackett copula for which the Spearman
# rank correlation coefficient is rho.
theta = bivariate_plackett_theta(rho)

print(f"Estimated theta: {theta}")

seed = 349289898325983983
rng = np.random.default_rng(seed)

psample = bivariate_plackett_sample(theta, m, random=rng)

# Weibull parameters
k = 1.6
scale = 4.0
# Convert uniform samples in psample to samples from the Weibull
# distribution by the inverse transform method.  (This changes
# the Pearson correlation, but not the Spearman correlation.)
wbl0, wbl1 = weibull_min.ppf(psample, k, scale=scale).T

# wbl0 and wbl1 are the correlated Weibull samples.

print("Desired Spearman correlation:", rho)
print("Sample Spearman correlation: ", spearmanr(wbl0, wbl1)[0])
print("Sample Pearson correlation:  ", pearsonr(wbl0, wbl1)[0])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plots
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

plt.figure(1)
plt.plot(wbl0, wbl1, '.', alpha=0.2)
plt.axis('equal')
plt.grid()
plt.title(f'Scatter plot of {m} samples from the bivariate distribution')
plt.savefig('plackett_copula_figure1.png')

plt.figure(2)
nbins = 40
plt.subplot(2, 1, 1)
t = np.linspace((k < 1)*1e-8, max(wbl0.max(), wbl1.max()), 500)
plt.hist(wbl0, bins=nbins, density=True, alpha=0.5)
plt.plot(t, weibull_min.pdf(t, k, scale=scale), 'k', alpha=0.7)
plt.grid()
plt.title(f'Marginal distributions\nWeibull parameters: k={k:g}, scale={scale:g}')
plt.subplot(2, 1, 2)
plt.hist(wbl1, bins=nbins, density=True, alpha=0.5)
plt.plot(t, weibull_min.pdf(t, k, scale=scale), 'k', alpha=0.7)
plt.grid()

# plt.show()
plt.savefig('plackett_copula_figure2.png')
