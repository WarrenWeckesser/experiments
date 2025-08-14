import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt('out', names=True)
names = data.dtype.names
xlabel = names[0]
ls = "-."
selection = [
    "invcdf1",
    "invcdf2",
    "invcdf3a",
    "invcdf3b",
#    "invcdft3",
#    "invcdft9",
#    "invcdft12",
#    "invcdft15",
#    "invcdfx",
     "invcdftp",
]
fig, ax = plt.subplots(len(selection), 1)
for k, name in enumerate(selection):
    # if ls == "-.":
    #     ls = ":"
    # else:
    #     ls = "-."
    ax[k].plot(data[xlabel], data[name], linewidth=1.5, alpha=0.75, label=name)
    ax[k].legend(framealpha=1, shadow=True)
    ax[k].semilogy()
    ax[k].set_ylim(1e-18, 5e-13)
    ax[k].set_xlim(-0.05, 1.05)
    ax[k].grid(True)
#plt.semilogy()
#plt.xlabel(xlabel)
fig.tight_layout()
plt.show()
