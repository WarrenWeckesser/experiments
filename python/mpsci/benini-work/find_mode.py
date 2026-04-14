import sympy
import numpy as np
import matplotlib.pyplot as plt


x, a, b, s = sympy.symbols('x a b s')
# PDF:
f = sympy.exp(-a*sympy.log(x/s) - b*sympy.log(x/s)**2) * (a/x + 2*b*sympy.log(x/s)/x)
df = sympy.diff(f, x)
mode = sympy.solve(df, x)
print(f"mode: {mode}")

pdf = sympy.lambdify((x, s, a, b), f, "numpy")

aa = 10
bb = 55
ss = 1.0
xx = np.linspace(ss, ss + 2.5, 1000)
yy = pdf(xx, ss, aa, bb)

mode0 = sympy.lambdify((s, a, b), mode[0], "numpy")
mode1 = sympy.lambdify((s, a, b), mode[1], "numpy")
m0 = mode0(ss, aa, bb)
m1 = mode1(ss, aa, bb)

print(f"{m0 = }")
print(f"{m1 = }")

if m0 >= ss:
    print("using m0")
    m = m0
elif m1 >= ss:
    print("using m1")
    m = m0
else:
    print("mode at ss")
    m = ss

plt.plot(xx, yy)
plt.plot(m, pdf(m, ss, aa, bb), '.')
plt.show()
