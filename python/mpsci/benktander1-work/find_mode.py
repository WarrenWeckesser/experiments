import sympy
import numpy as np
import matplotlib.pyplot as plt


x, a, b = sympy.symbols('x a b')
# PDF:
t = a + 2 * b * sympy.log(x)
f = (t / a * (1 + t) - 2 * b / a) * x**(-(2 + a + b * sympy.log(x)))
df = sympy.diff(f, x)
mode = sympy.solve(df, x)
print(f"mode: {mode}")
if len(mode) != 1:
    print(f"({len(mode)} formulas)")

pdf = sympy.lambdify((x, a, b), f, "numpy")

aa = 12
bb = 30
xx = np.linspace(1, 1.5, 1000)
yy = pdf(xx, aa, bb)

mode0 = sympy.lambdify((a, b), mode[0], "numpy")
mode1 = sympy.lambdify((a, b), mode[1], "numpy")
mode2 = sympy.lambdify((a, b), mode[2], "numpy")
m0 = mode0(aa, bb)
m1 = mode1(aa, bb)
m2 = mode2(aa, bb)

print(f"{m0 = }")
print(f"{m1 = }")
print(f"{m2 = }")

m = max(1, m2)
plt.plot(xx, yy)
plt.plot(m, pdf(m, aa, bb), '.')
plt.grid(True)
plt.show()
