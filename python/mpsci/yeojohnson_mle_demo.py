from mpmath import mp
from mpsci.stats import yeojohnson_mle, yeojohnson_llf
import matplotlib.pyplot as plt


mp.dps = 50

# x = [1, 2, 3, 5, 8, 13]
# lam0 = -0.1

x = [2003.0, 1950.0, 1997.0, 2000.0, 2009.0,
     2009.0, 1980.0, 1999.0, 2007.0, 1991.0]
lam0 = 100

lam = yeojohnson_mle(x, lam0=lam0)
print(f'{lam = }')

lam_vals = mp.linspace(0.95*lam, 1.05*lam, 500)
llf = [yeojohnson_llf(t, x) for t in lam_vals]

plt.plot(lam_vals, llf)
plt.plot(lam, yeojohnson_llf(lam, x), 'kd')
plt.xlabel(r'$\lambda$')
plt.ylabel('log-likelihood')
plt.title('Yeo-Johnson log-likelihood function\n'
          f'$\\hat\\lambda$ = {float(lam):12.8f}')
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()
