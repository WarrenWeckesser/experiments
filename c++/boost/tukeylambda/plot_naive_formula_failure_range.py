import numpy as np
from scipy.special import logit
import matplotlib.pyplot as plt


p, lam_low, lam_high = np.loadtxt('naive_formula_failure_range.dat', unpack=True)

plt.plot(p, abs(lam_low), label="-low")
plt.plot(p, lam_high, label="high")
pfine = np.linspace(1e-10, 0.9999999999, 800)
plt.plot(pfine, 0.07/abs(logit(pfine)))

plt.grid(alpha=0.5)
plt.semilogy()

plt.show()
