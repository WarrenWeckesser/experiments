
import numpy as np
import matplotlib.pyplot as plt


print("Reading...")
values = np.loadtxt('relerr.txt')

kmax = np.argmax(np.abs(values[:, 2]))

print(f"Maximum relative error is {values[kmax, 2]} "
      f"at {values[kmax, :2].tolist()} (index {kmax})")

print("Plotting...")
gridsize = 150
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1, 1, 1)
ax1.hexbin(values[:, 0], values[:, 1], C=values[:, 2], gridsize=gridsize)
ax1.set_xlabel('x')
ax1.set_ylabel('y')

k_nonzero_relerr = np.where(values[:, 2] != 0)[0]
print(f"{len(k_nonzero_relerr)} of {len(values)} have nonzero relative error")
values_nzerr = values[k_nonzero_relerr]
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(1, 1, 1)
ax2.hist(np.log10(values_nzerr[:, 2]), bins=200)
ax2.set_xlabel('log10(relative_error)')

plt.show()
