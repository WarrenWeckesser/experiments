from mpmath import mp
from mpsci.stats import mean as _mean
from mpsci.distributions import binomial, Initial
import matplotlib.pyplot as plt


mp.dps = 100

#x = [8, 6, 6, 8, 8, 7, 9, 11, 8, 7, 10, 10, 8, 10, 10, 9, 9,
#     9, 9, 10, 7, 8, 8, 8, 10, 6, 10, 7, 9, 8, 3, 6]
#x = [0, 1, 4, 3, 2, 6]
#x = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 5]
#x = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4]
#x = [12,  9, 14, 12,  9, 10, 14,  9,  9,  8, 15, 10]
#x = [0, 3, 1, 1, 2, 3, 3, 0, 2, 2, 1, 3]
x = [0, 1, 3, 5, 4, 1, 1, 1, 0, 2, 4, 0]
counts = None

m = _mean(x)

def mle_n_eqn(n):
    p1 = m/n
    return (-_mean([mp.digamma(n - xi + 1) for xi in x],
                    weights=counts)
            + mp.digamma(n + 1) + mp.log1p(-p1))


#n = mp.linspace(max(x), max(x) + 5, 1000)
n = mp.linspace(m, max(x) + 50, 1200)
y = [mle_n_eqn(n1) for n1 in n]

plt.plot(n, y)
plt.grid()
plt.show()
