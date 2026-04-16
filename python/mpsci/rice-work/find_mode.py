import sympy
from sympy import init_printing
import numpy as np
import matplotlib.pyplot as plt


init_printing()

x, nu, sigma = sympy.symbols('x nu sigma')
# PDF:
f = (x / sigma**2) * sympy.exp(-(x**2 + nu**2)/(2*sigma**2)) * sympy.besseli(0, x*nu/sigma**2)
df = sympy.diff(f, x)
df = sympy.simplify(df)

print("Derivative of the PDF of the Rice distribution:")
sympy.pprint(df)

