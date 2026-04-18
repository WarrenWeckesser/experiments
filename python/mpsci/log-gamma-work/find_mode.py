import sympy
from sympy import init_printing
import numpy as np
import matplotlib.pyplot as plt


init_printing()

x, k, theta = sympy.symbols('x k theta')
# PDF:
z = x/theta
f = sympy.exp(k*z - sympy.exp(z)) / sympy.gamma(k) / theta

df = sympy.diff(f, x)
df = sympy.simplify(df)

print("Derivative of the PDF of the gamma-Gompertz distribution:")
sympy.pprint(df)

print()
mode = sympy.solve(df, x)
print(f"mode: {mode}")
