# Generate Figure 20.3 or 20.5 of Abramowitz and Stegun.

import numpy as np
from scipy.special import mathieu_sem
import matplotlib.pyplot as plt


x = np.linspace(0, 90, 1000)

# Figure 20.3: q = 1
# Figure 20.5: q = 10
q = 1
ce1, dce1 = mathieu_sem(1, q, x)
ce2, dce2 = mathieu_sem(2, q, x)
ce3, dce3 = mathieu_sem(3, q, x)
ce4, dce4 = mathieu_sem(4, q, x)
ce5, dce5 = mathieu_sem(5, q, x)

dash_scale = 1.25
dashes2 = (dash_scale*np.array([2.5, 0.75])).tolist()
dashes3 = (dash_scale*np.array([5, 0.75])).tolist()
dashes4 = (dash_scale*np.array([6, 0.75, 2.0, 0.75])).tolist()
dashes5 = (dash_scale*np.array([6, 0.75, 0.75, 0.75])).tolist()
lw = 1.5
plt.figure(figsize=(8, 6))
plt.plot(x, ce1, label=r'se$_1$', lw=lw)
plt.plot(x, ce2, label=r'se$_2$', lw=lw,
         dashes=dashes2)
plt.plot(x, ce3, label=r'se$_3$', lw=lw,
         dashes=dashes3)
plt.plot(x, ce4, label=r'se$_4$', lw=lw,
         dashes=dashes4)
plt.plot(x, ce5, label=r'se$_5$', lw=lw,
         dashes=dashes5)

plt.legend(framealpha=1, shadow=True)

xticks = range(0, 100, 10)
xaxis = plt.gca().xaxis
xaxis.set_ticks(xticks)
labels = ["$" + str(n) + r"^{\circ}$" for n in xticks]
xaxis.set_ticklabels(labels)

plt.grid(alpha=0.25)
plt.title("Odd Periodic Mathieu Functions\nOrders 1 - 5, q = %d" % q)

plt.show()
