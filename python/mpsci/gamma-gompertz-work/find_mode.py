import sympy
from sympy import init_printing
import numpy as np
import matplotlib.pyplot as plt


init_printing()

x, c, beta, scale = sympy.symbols('x c beta scale')
# PDF:
z = x/scale
num = c * sympy.exp(z) * beta**c
den = scale * (beta + (sympy.exp(z) - 1))**(c + 1)
f = num / den
df = sympy.diff(f, x)
df = sympy.simplify(df)

print("Derivative of the PDF of the gamma-Gompertz distribution:")
sympy.pprint(df)

print()
mode = sympy.solve(df, x)
print(f"mode: {mode}")

