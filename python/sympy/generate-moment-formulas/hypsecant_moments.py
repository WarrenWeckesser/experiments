import sympy
from genmom import generate_moments


t = sympy.symbols('t')

mgf = sympy.sec(t)
n = 14
moments = generate_moments(mgf, t, n)

print("Moments about 0 for the hyperbolic secant distribution "
      "(wikipedia parametrization)")
for k in range(1, n + 1):
    print(f'm{k}: {moments[k-1]}')
