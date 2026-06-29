# Generate Figure 20.2 or 20.4 of Abramowitz and Stegun.

import numpy as np
from scipy.special import mathieu_cem
import matplotlib.pyplot as plt


x = np.linspace(0, 90, 1000)

# Figure 20.2: q = 1
# Figure 20.4: q = 10
q = 1
ce0, dce0 = mathieu_cem(0, q, x)
ce1, dce1 = mathieu_cem(1, q, x)
ce2, dce2 = mathieu_cem(2, q, x)
ce3, dce3 = mathieu_cem(3, q, x)
ce4, dce4 = mathieu_cem(4, q, x)
ce5, dce5 = mathieu_cem(5, q, x)

dash_scale = 1.25
dashes1 = (dash_scale*np.array([2.5, 0.75])).tolist()
dashes2 = (dash_scale*np.array([5, 0.75, 1.5, 0.75])).tolist()
dashes3 = (dash_scale*np.array([6, 0.75, 0.75, 0.75])).tolist()
dashes4 = (dash_scale*np.array([6, 0.75])).tolist()
dashes5 = (dash_scale*np.array([6, 0.75, 0.75, 0.75, 0.75, 0.75])).tolist()
lw = 1.5
plt.figure(figsize=(8, 6))
plt.plot(x, ce0, label=r'ce$_0$', lw=lw)
plt.plot(x, ce1, label=r'ce$_1$', lw=lw,
         dashes=dashes1)
plt.plot(x, ce2, label=r'ce$_2$', lw=lw,
         dashes=dashes2)
plt.plot(x, ce3, label=r'ce$_3$', lw=lw,
         dashes=dashes3)
plt.plot(x, ce4, label=r'ce$_4$', lw=lw,
         dashes=dashes4)
plt.plot(x, ce5, label=r'ce$_5$', lw=lw,
         dashes=dashes5)

plt.legend(framealpha=1, shadow=True)

xticks = range(0, 100, 10)
xaxis = plt.gca().xaxis
xaxis.set_ticks(xticks)
labels = ["$" + str(n) + r"^{\circ}$" for n in xticks]
xaxis.set_ticklabels(labels)

plt.grid(alpha=0.25)
plt.title("Even Periodic Mathieu Functions\nOrders 0 - 5, q = %d" % q)

# plt.show()
plt.savefig('images/even_periodic_mathieu_functions_plot.svg')
