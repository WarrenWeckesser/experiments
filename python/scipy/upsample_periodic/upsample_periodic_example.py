
import numpy as np
import matplotlib.pyplot as plt
from upsample_periodic import upsample_periodic


x = np.array([2.25, 1.0, -1.5, -1.25, 0.25, 0.25])
dt = 0.5
tx = np.arange(len(x))*dt

n1 = 15
y1, dt_out1 = upsample_periodic(x, n1, dt=dt)
ty1 = np.arange(len(y1))*dt_out1

n2 = 8*len(x)
y2, dt_out2 = upsample_periodic(x, n2, dt=dt)
ty2 = np.arange(len(y2))*dt_out2

T = len(x)*dt

plt.plot(tx, x, 'ko', alpha=0.75, label=f'data ({len(x)} samples, dt={dt})')
plt.plot(ty1, y1, 'o-', alpha=0.6, markersize=4,
         label=f'periodic upsample (to n={n1})')
plt.plot(ty2, y2, 'o-', alpha=0.6, markersize=2.5,
         label=f'periodic upsample (8*, to n={n2})')

plt.grid(True)
plt.xlabel('t')
plt.legend(framealpha=1, shadow=True)

plt.show()
# plt.savefig('upsample_periodic_example.svg')
