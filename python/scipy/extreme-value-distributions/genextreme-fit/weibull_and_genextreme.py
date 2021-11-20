
import numpy as np
from scipy.stats import genextreme, weibull_max
import matplotlib.pyplot as plt


def gev_to_wmax(xi, mu, sigma):
    """
    Convert generalized extreme value parameters to Weibull max parameters.
    """
    alpha = 1/xi
    a = sigma/xi
    b = mu + a
    return alpha, b, a


data = np.array([37.50, 46.79, 48.30, 46.04, 43.40, 39.25, 38.49, 49.51,
                 40.38, 36.98, 40.00, 38.49, 37.74, 47.92, 44.53, 44.91,
                 44.91, 40.00, 41.51, 47.92, 36.98, 43.40, 42.26, 41.89,
                 38.87, 43.02, 39.25, 40.38, 42.64, 36.98, 44.15, 44.91,
                 43.40, 49.81, 38.87, 40.00, 52.45, 53.13, 47.92, 52.45,
                 44.91, 29.54, 27.13, 35.60, 45.34, 43.37, 54.15, 42.77,
                 42.88, 44.26, 27.14, 39.31, 24.80, 16.62, 30.30, 36.39,
                 28.60, 28.53, 35.84, 31.10, 34.55, 52.65, 48.81, 43.42,
                 52.49, 38.00, 38.65, 34.54, 37.70, 38.11, 43.05, 29.95,
                 32.48, 24.63, 35.33, 41.34])


# Fit genextreme to the data.
ge_c, ge_loc, ge_scale = genextreme.fit(data)

# Convert the genextreme parameters to the weibull_max parameters.
wmax = gev_to_wmax(ge_c, ge_loc, ge_scale)

# Fit weibull_max to the data.  In theory, the parameters in wmax_fit
# should be the same as those in wmax.
# (The added constants are just to ensure that the optimization performed
# by the fit method does a little work.)
wm_c, wm_loc, wm_scale = weibull_max.fit(data, wmax[0]+0.1, loc=wmax[1]+0.5,
                                         scale=wmax[2]+0.05)

print("                         shape      loc        scale")
print(f"genextreme.fit():       {ge_c:.6f}, {ge_loc:.6f}, {ge_scale:9.6f}")
print(f"convert to weibull_max: {wmax[0]:.6f}, {wmax[1]:.6f}, {wmax[2]:9.6f}")
print(f"weibull_max.fit():      {wm_c:.6f}, {wm_loc:.6f}, {wm_scale:9.6f}")


plt.hist(data, alpha=0.5, density=True, bins=20,
         label='normalized histogram')
x = np.linspace(0, 60, 1001)
plt.plot(x, genextreme.pdf(x, ge_c, loc=ge_loc, scale=ge_scale),
         lw=3, alpha=0.6,
         label='genextreme.fit()')
# plt.plot(x, weibull_max.pdf(x, *wmax), lw=2,
#          label='weibull_max')
plt.plot(x, weibull_max.pdf(x, wm_c, loc=wm_loc, scale=wm_scale), 'k--', lw=1,
         label='weibull_max.fit()')
plt.legend(loc='best')
plt.text(-1, 0.045,
         "                     shape    loc      scale\n"
         f"genextreme.fit():   {ge_c:.5f} {ge_loc:.5f} {ge_scale:8.5f}\n"
         f"convert to weibull: {wmax[0]:.5f} {wmax[1]:.5f} {wmax[2]:8.5f}\n"
         f"weibull_max.fit():  {wm_c:.5f} {wm_loc:.5f} {wm_scale:8.5f}",
         fontfamily='monospace', fontsize=7,
         bbox=dict(facecolor='gray', alpha=0.4))
plt.title('Fit genextreme and weibull_max to the same data')
# plt.show()
plt.savefig('weibull_and_genextreme.png', dpi=125)
