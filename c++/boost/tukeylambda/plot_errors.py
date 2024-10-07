import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt('out', names=True)
names = data.dtype.names
xlabel = names[0]
ls = "-."
selection = ["invcdf1", "invcdf3"]
# selection = ["invcdft3", "invcdft9", "invcdft12", "invcdft15", "invcdfx"]
for name in selection:
    if ls == "-.":
        ls = ":"
    else:
        ls = "-."
    plt.plot(data[xlabel], data[name], linewidth=2.5, linestyle=ls, alpha=0.65, label=name)
plt.legend(framealpha=1, shadow=True)
plt.grid()
plt.semilogy()
plt.show()
