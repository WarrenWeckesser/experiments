# Benini distribution, MLE scratch work.

import sympy as sp
from sympy import init_printing
# import numpy as np
# import matplotlib.pyplot as plt

init_printing(use_unicode=True)


X = sp.IndexedBase('X')
i, n = sp.symbols('i n', integer=True)
x, a, b, s = sp.symbols('x a b s')
# PDF:
f = sp.exp(-a*sp.log(x/s) - b*sp.log(x/s)**2) * (a/x + 2*b*sp.log(x/s)/x)

logf = sp.log(f)
logf = sp.expand_log(logf, force=True)

# print("logf (after expand_log):")
# sp.pprint(logf)

loglik = sp.Sum(logf.subs(x, X[i]), (i, 1, n))

print("log-likelihood:")
sp.pprint(loglik)

adiff = loglik.diff(a)
adiff = sp.simplify(adiff)

print("adiff:")
sp.pprint(adiff)

bdiff = loglik.diff(b)
bdiff = sp.simplify(bdiff)

print("bdiff:")
sp.pprint(bdiff)


sdiff = loglik.diff(s)
sdiff = sp.simplify(sdiff)

print("sdiff:")
sp.pprint(sdiff)

# pdf = sympy.lambdify((x, s, a, b), f, "numpy")

# aa = 10
# bb = 55
# ss = 1.0
# xx = np.linspace(ss, ss + 2.5, 1000)
# yy = pdf(xx, ss, aa, bb)

# mode0 = sympy.lambdify((s, a, b), mode[0], "numpy")
# mode1 = sympy.lambdify((s, a, b), mode[1], "numpy")
# m0 = mode0(ss, aa, bb)
# m1 = mode1(ss, aa, bb)

# print(f"{m0 = }")
# print(f"{m1 = }")

# if m0 >= ss:
#     print("using m0")
#     m = m0
# elif m1 >= ss:
#     print("using m1")
#     m = m0
# else:
#     print("mode at ss")
#     m = ss

# plt.plot(xx, yy)
# plt.plot(m, pdf(m, ss, aa, bb), '.')
# plt.show()
