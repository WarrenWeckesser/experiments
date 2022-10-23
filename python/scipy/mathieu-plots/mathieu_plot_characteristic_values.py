# Generate Figure 20.1 of Abramowitz and Stegun.

import numpy as np
from scipy.special import mathieu_a, mathieu_b
import matplotlib.pyplot as plt


clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']

q = np.linspace(0, 15, 151)

plt.figure(figsize=(5.0, 5.5))
for order in range(0, 6):
    a = mathieu_a(order, q)
    if order > 0:
        b = mathieu_b(order, q)
    if order == 1:
        alabel = '$a_r(q)$'
        blabel = '$b_r(q)$'
    else:
        alabel = None
        blabel = None
    plt.plot(q, a, color=clrs[0], label=alabel)
    if order > 0:
        plt.plot(q, b, '--', color=clrs[1], label=blabel)

plt.text(0.3, -1.5, '$a_0$')
plt.text(1.06, 0.0, '$b_1$')
plt.text(0.3, 2.1, '$a_1$')
plt.text(3.1, 3.9, '$b_2$')
plt.text(2.8, 6.9, '$a_2$')
plt.text(5.5, 9.6, '$b_3$')
plt.text(6.2, 14, '$a_3$')
plt.text(10.0, 17.8, '$b_4$')
plt.text(10.9, 23, '$a_4$')
plt.text(12.2, 26.0, '$b_5$')
plt.text(12.5, 31, '$a_5$')

yax = plt.gca().yaxis
yticks = range(-24, 34, 4)
yax.set_ticks(yticks)
plt.xlim(q[0], q[-1])
plt.xlabel('$q$')
plt.legend(framealpha=1, shadow=True)

plt.grid(alpha=0.25)
plt.title("Mathieu Equation Characteristic Values")
# plt.show()
plt.savefig('images/mathieu_plot_characteristic_values.svg', dpi=125)
