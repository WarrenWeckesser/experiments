import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zipfian

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

plt.plot(xx, pdf,
         label='Zipfian "PDF"')

plt.grid(visible=True)
plt.xlabel('x')
plt.title(f'Zipfian "PDF" (a={a}, n={n})')
plt.savefig('zipfian_pdf.png')

# Figure 3. Nonnormalized PDF


nnpdf = h(xx, a, n)

plt.figure(figsize=figsize)

plt.plot(xx, nnpdf,
         label='Zipfian Nonnormalized PDF')

plt.grid(visible=True)
plt.xlabel('x')
plt.title(f'Zipfian PDF (a={a}, n={n})')
plt.savefig('zipfian_nnpdf.png')