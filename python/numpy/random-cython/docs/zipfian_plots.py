import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zipfian
from scipy.special import boxcox
from scipy.special._ufuncs import _gen_harmonic


def f(x, a, n):
    # Piecewise constant Zipfian PDF.
    mask = (x >= 1) & (x < n + 1)
    out = np.zeros_like(x)
    out[mask] = np.trunc(x[mask])**-a
    return out / _gen_harmonic(n, a)


def h(x, a, n):
    # This is the target "histogram function" i.e. the nonnormalized PMF,
    # expanded to be a function of the continuous variable x.
    mask = (x >= 1) & (x < n + 1)
    out = np.zeros_like(x)
    out[mask] = np.trunc(x[mask])**-a
    return out


def g(x, a, n):
    # PDF of the dominating distribution.
    out = np.zeros(len(x))
    out[(1 <= x) & (x <= 2)] = 1.0
    mask = (2 < x) & (x < n + 1)
    out[mask] = (x[mask] - 1)**-a
    return out/(boxcox(n, 1 - a) + 1)


def G(x, a, n):
    # Currently, the only call of this function used in the
    # rejection method is G(n + 1, a, n).
    x = np.atleast_1d(x)
    out = np.zeros(len(x))
    out[x >= n + 1] = boxcox(n, 1 - a) + 1
    mask1 = (1 <= x) & (x <= 2)
    out[mask1] = x[mask1] - 1
    mask = (2 < x) & (x < n + 1)
    out[mask] = boxcox(x[mask] - 1, 1 - a) + 1
    return out/(boxcox(n, 1 - a) + 1)


def M(a, n):
    return (boxcox(n, 1 - a) + 1) / _gen_harmonic(n, a)

a = 0.95
n = 7
k = np.arange(1, n + 1)
pmf = zipfian.pmf(k, a, n)

figsize = (7, 4)

# Figure 1.

plt.figure(figsize=figsize)

plt.stem(k, pmf, basefmt=" ")

plt.grid(visible=True)
plt.xlabel('k')
plt.title(f'Zipfian PMF p(k, a={a}, n={n})')
plt.savefig('zipfian_pmf.png')

# Figure 2.  The PDF

xx = np.linspace(np.nextafter(1.0, 0), n + 1, 8000)
pdf = f(xx, a, n)

plt.figure(figsize=figsize)

plt.plot(xx, pdf)

plt.grid(visible=True)
plt.xlabel('x')
plt.title(f'Zipfian PDF f(x, a={a}, n={n})')
plt.savefig('zipfian_pdf.png')

# Figure 3. PDF and the scaled dominating PDF.

dom = M(a, n) * g(xx, a, n)

plt.figure(figsize=figsize)

plt.plot(xx, pdf, label='PDF $f(x, a, n)$')
plt.plot(xx, dom, '--', label='$M(a, n) g(x, a, n)$')

plt.legend(shadow=True, framealpha=1)
plt.grid(visible=True)
plt.xlabel('x')
plt.title(f'Zipfian PDF and Scaled Dominating PDF (a={a}, n={n})')
plt.savefig('zipfian_pdf_and_dom.png')

# Figure 4. G(x, a, n)

plt.figure(figsize=figsize)

plt.plot(xx, G(xx, a, n), 'k',
         label='G(x, a, n)')
plt.plot(1, 0, 'k.')
plt.plot(2, G(2, a, n), 'k.')
plt.title('G(x, a, n), the CDF of the dominating distribution')


plt.grid(visible=True)
plt.xlabel('x')
plt.title(f'G(x, a={a}, n={n})')

plt.savefig('zipfian_dom_cdf.png')
