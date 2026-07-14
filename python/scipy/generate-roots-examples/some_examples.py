import numpy as np

#
# roots_hermite
#

from scipy.special import roots_hermite, eval_hermite, erfc


def f(x):
    return 1.0 / (x**2 + 1.0)


exact = np.pi * np.e * erfc(1)

print("    n     quadrature       rel. error")
for n in [5, 10, 20, 40, 80]:
    x, weights = roots_hermite(n)
    q = weights @ f(x)
    relerr = abs((q - exact) / exact)
    print(f"{n:5} {q:<20}  {relerr:9.3e}")


#
# roots_hermitenorm
#

from scipy.special import roots_hermitenorm, eval_hermitenorm, erfc


def f(x):
    return 1.0 / (x**2 + 1.0)


exact = np.pi * np.exp(0.5) * erfc(1/np.sqrt(2))

print("    n     quadrature       rel. error")
for n in [5, 10, 20, 40, 80]:
    x, weights = roots_hermitenorm(n)
    q = weights @ f(x)
    relerr = abs((q - exact) / exact)
    print(f"{n:5} {q:<20}  {relerr:9.3e}")


#
# roots_sh_legendre
#

from scipy.special import roots_sh_legendre, eval_sh_legendre

def f(x):
    return x + 1.0/x

a = 0.5
b = 2.0

# Exact integral
exact = 15/8 + 2*np.log(2)

print("    n     quadrature       rel. error")
for n in [5, 10, 20]:
    x, weights = roots_sh_legendre(n)
    q = (b - a) * (weights @ f(a + x*(b - a)))
    relerr = abs((q - exact) / exact)
    print(f"{n:5} {q:<20}  {relerr:9.3e}")
