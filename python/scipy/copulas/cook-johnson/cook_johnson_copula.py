
import numpy as np
from scipy.stats import weibull_min, spearmanr, pearsonr
import matplotlib.pyplot as plt


# Generate sample from the Cook-Johnson copula with marginal distribution
# from the Weibull distribution
# From https://www.frontiersin.org/articles/10.3389/fams.2021.642210/full
# in the section "Simulation From the Cook–Johnson Copula"

def bivariate_cook_johnson_sample(alpha, m, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    y = rng.exponential(size=(2, m))
    z = rng.gamma(1/alpha, size=m)

    u = (1 + y/z)**(-1/alpha)
    return u


alpha = 0.5
m = 6000

u = bivariate_cook_johnson_sample(alpha, m)

# Weibull parameters
k = 1.6
scale = 4.0
w = weibull_min.ppf(u, k, scale=scale)

print("Spearman: ", spearmanr(w[0], w[1])[0])
print("Pearson:  ", pearsonr(w[0], w[1])[0])

plt.figure(1)
plt.plot(w[0], w[1], '.', alpha=0.2)
plt.axis('equal')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sample from the Cook-Johnson copula with Weibull marginals')

plt.savefig('images/scatter-plot.png')

plt.figure(2)
t = np.linspace(0, 1.05*w.max(), 500)
p = weibull_min.pdf(t, k, scale=scale)

plt.subplot(2, 1, 1)
plt.hist(w[0], bins=75, density=True, alpha=0.4)
plt.plot(t, p, 'k', alpha=0.5)
plt.title('x marginal distribution')

plt.subplot(2, 1, 2)
plt.hist(w[1], bins=75, density=True, alpha=0.4)
plt.plot(t, p, 'k', alpha=0.5)
plt.title('y marginal distribution')

plt.tight_layout()

plt.savefig('images/marginal-distributions.png')

# plt.show()
