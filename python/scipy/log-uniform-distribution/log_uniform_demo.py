
import numpy as np
from scipy.stats import loguniform
import matplotlib.pyplot as plt


# Recreate (more or less) the PDF and CDF plots shown in the wikipedia article.

n = 1000
x = np.linspace(0, 7, n)
pdf1 = loguniform.pdf(x, 1, 4)
pdf2 = loguniform.pdf(x, 2, 6)

cdf1 = loguniform.cdf(x, 1, 4)
cdf2 = loguniform.cdf(x, 2, 6)

plt.figure(1)
plt.plot(x, pdf1, 'y', label='a = 1, b = 4')
plt.plot(x, pdf2, 'm', label='a = 2, b = 6')
plt.legend(framealpha=1, shadow=True)
plt.grid(True)
plt.title('Probability density function')
plt.show()


plt.figure(2)
plt.plot(x, cdf1, 'y', label='a = 1, b = 4')
plt.plot(x, cdf2, 'm', label='a = 2, b = 6')
plt.legend(framealpha=1, shadow=True)
plt.grid(True)
plt.title('Cumulative distribution function')
plt.show()
