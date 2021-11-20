

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme


x = np.genfromtxt('data/portpirie.csv',
                  dtype=[('year', int), ('max', float)])
years = x['year']
maxima = x['max']


# Note that Coles' shape parameter xi is the opposite sign of scipy's c.
c, mu, sigma = genextreme.fit(maxima)
print("Fit of generalized extreme distribution")
print(f"Shape c:     {c:7.4f} (note that Coles' ξ = -c)")
print(f"Loc mu:      {mu:7.4f}")
print(f"Scale sigma: {sigma:7.4f}")


plt.figure(1)
plt.plot(years, maxima, '.')
plt.xlabel('Year')
plt.ylabel('Sea-level (meters)')
plt.title('Annual maximum sea levels at Port Pirie, South Australia\n'
          '(Figure 1.1 of Coles (2001))')

plt.figure(2)
plt.hist(maxima, bins=24, density=True, label='normalized histogram')
xx = np.linspace(0.95*maxima.min(), 1.05*maxima.max(), 200)
plt.plot(xx, genextreme.pdf(xx, c, loc=mu, scale=sigma),
         label='PDF of fitted genextreme')
plt.text(maxima.min() + 0.6*maxima.ptp(), 2.1,
         f'genextreme fit:\n  c = -ξ = {c:.5f}\n'
         f'  mu     = {mu:.5f}\n  sigma  = {sigma:.5f}',
         fontfamily='monospace',
         bbox=dict(facecolor='gray', alpha=0.4))
plt.legend()
plt.title('Maximum likelihood fit of genextreme to the Port Pirie data\n'
          '(Coles Example 3.4.1)')
# plt.show()
plt.savefig('plot_portpirie.png', dpi=125)
