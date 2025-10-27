import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zipfian
from scipy.special import boxcox

def zipfian_pdf(x, a, n):
    # Treat the discrete distribution as a piecewise constant
    # continuous distribution with support [0.5, n + 0.5]
    mask = (x > 0.5) & (x < n + 0.5)
    out = np.zeros_like(x)
    out[mask] = zipfian.pmf(np.round(x[mask]), a, n)
    # out[mask] = np.round(x[mask])**-a
    return out


def h(x, a, n):
    # This is the target "histogram function" i.e. the nonnormalized PMF,
    # expanded to be a function of the continuous variable x.
    mask = (x > 0.5) & (x < n + 0.5)
    out = np.zeros_like(x)
    out[mask] = np.round(x[mask])**-a
    return out

def g(x, a, n):
    # g is the "hat" function
    out = np.zeros(len(x))
    out[(0.5 <= x) & (x <= 1.5)] = 1.0
    mask = (1.5 < x) & (x < n + 0.5)
    out[mask] = (x[mask] - 0.5)**-a
    return out


def G(x, a, n):
    # Currently, the only call of this function used in the
    # rejection method is G(n + 0.5, a, n).
    x = np.atleast_1d(x)
    out = np.zeros(len(x))
    out[x >= n + 0.5] = boxcox(n, 1 - a) + 1
    mask1 = (0.5 <= x) & (x <= 1.5)
    out[mask1] = x[mask1] - 0.5
    mask = (1.5 < x) & (x < n + 0.5)
    out[mask] = boxcox(x[mask] - 0.5, 1 - a) + 1
    return out


a = 0.95
n = 7
k = np.arange(1, n + 1)
pmf = zipfian.pmf(k, a, n)

figsize = (7.5, 4.5)

# Figure 1.

plt.figure(figsize=figsize)

plt.plot(k, pmf, 'o', ms=3.5)

plt.grid(visible=True)
plt.xlabel('x')
plt.title(f'Zipfian PMF (a={a}, n={n})')
plt.savefig('zipfian_pmf.png')

# Figure 2.  The PDF

xx = np.linspace(0.5, n + 0.5, 8000)
pdf = zipfian_pdf(xx, a, n)

plt.figure(figsize=figsize)

plt.plot(xx, pdf)

plt.grid(visible=True)
plt.xlabel('x')
plt.title(f'Zipfian PDF (a={a}, n={n})')
plt.savefig('zipfian_pdf.png')

# Figure 3. Nonnormalized PDF

nnpdf = h(xx, a, n)

plt.figure(figsize=figsize)

plt.plot(xx, nnpdf)

plt.grid(visible=True)
plt.xlabel('x')
plt.title(f'Zipfian Nonnormalized PDF (a={a}, n={n})')
plt.savefig('zipfian_nnpdf.png')

# Figure 4. Nonnormalized PDF and the dominating function.

dom = g(xx, a, n)

plt.figure(figsize=figsize)

plt.plot(xx, nnpdf, label='Nonnormalized PDF $h(x, a, n)$')
plt.plot(xx, dom, '--', label='Dominating function $g(x, a, n)$')

plt.legend(shadow=True, framealpha=1)
plt.grid(visible=True)
plt.xlabel('x')
plt.title(f'Zipfian Nonnormalized PDF and Dominating Function (a={a}, n={n})')
plt.savefig('zipfian_nnpdf_and_dom.png')

# Figure 5. G(x, a, n)

plt.figure(figsize=figsize)

plt.plot(xx, G(xx, a, n), 'k',
         label='G(x, a, n)\nintegral of the dominating nonnormalized PDF g(x, a, n)')
plt.plot(0.5, 0, 'k.')
plt.plot(1.5, G(1.5, a, n), 'k.')
maxG = G(n + 0.5, a, n)
plt.plot(n + 0.5, maxG, 'k.')
plt.axhline(maxG, linestyle=':', alpha=0.5,
            label='max G(x, a, n) = G(n + 0.5, a, n)')

plt.legend(shadow=True, framealpha=1)
plt.grid(visible=True)
plt.xlabel('x')
plt.title(f'Zipfian Rejection Method  (a={a}, n={n})\n'
          'G(x, a, n), the nonnormalized CDF of the dominating distribution')

plt.savefig('zipfian_dom_nncdf.png')
