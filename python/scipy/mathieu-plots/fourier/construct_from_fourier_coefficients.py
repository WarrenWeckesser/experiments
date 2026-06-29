import numpy as np
import matplotlib.pyplot as plt
from scipy.special import mathieu_even_coef, mathieu_cem

n = 7
q = 14

period, shift = (180.0, 0) if n % 2 == 0 else (360.0, 1)

# Use the Fourier coefficients to construct the periodic Mathieu function.
A = mathieu_even_coef(n, q)
x = np.linspace(0, period, 5000)
t = (np.pi/180)*x
k = np.arange(len(A)).reshape((-1, 1))
c = np.cos((2*k + shift)*t)

y = A @ c

plt.plot(x, y, linewidth=3.5, alpha=0.3, label="Fourier sum")
ce, dce = mathieu_cem(n, q, x)
plt.plot(x, ce, 'k--', alpha=0.5, label="mathieu_cem")
plt.grid(True)
plt.title(f'Mathieu Function $\\rm{{ce_{n}}}(x, {q})$')
plt.xlabel('x [degrees]')
plt.legend(framealpha=1, shadow=True)

# plt.show()
plt.savefig('mathieu_function_from_fourier.png')
