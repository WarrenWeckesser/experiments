
import numpy as np
from scipy.stats import binomtest, binom_test

halfminus = np.nextafter(0.5, 0)
halfplus = np.nextafter(0.5, 1)

p8a = 70**0.25 / (70**0.25 + 1)
p8aminus = np.nextafter(p8a, 0)
p8aplus = np.nextafter(p8a, 1)
p8b = 1 / (70**0.25 + 1)
p8bminus = np.nextafter(p8b, 0)
p8bplus = np.nextafter(p8b, 1)

p12minus = 0.07692307692
p12plus  = 0.07692307693

cases = [
   (0, 1, halfminus),
   (0, 1, 0.5),
   (0, 1, halfplus),
   (0, 1, 0.25),
   (0, 1, 0.75),
   (0, 1, 0.0),
   (0, 1, 1.0),
   (0, 7, 0.25),
   (1, 7, 0.25),
   (2, 7, 0.25),
   (0, 8, 0.0),
   (0, 8, 1.0),
   (4, 8, 0.0),
   (4, 8, p8a),
   (4, 8, p8aplus),
   (4, 8, p8aminus),
   (4, 8, p8b),
   (4, 8, p8bplus),
   (4, 8, p8bminus),
   (3, 8, 0.5),
   (4, 8, 0.5),
   (3, 8, halfminus),
   (4, 8, halfminus),
   (3, 9, 0.5),
   (4, 9, 0.5),
   (3, 9, halfminus),
   (4, 9, halfminus),
   (0, 3, 0.5),
   (1, 3, 0.5),
   (0, 4, 0.01),
   (1, 4, 0.01),
   (0, 12, p12minus),
   (1, 12, p12minus),
   (2, 12, p12minus),
   (0, 12, p12plus),
   (1, 12, p12plus),
   (2, 12, p12plus),
]

for k, (x, n, p) in enumerate(cases):
    pold = binom_test(x, n, p)
    pnew = binomtest(x, n, p).pvalue
    if pold != pnew:
        print(k, x, n, p)
    pold = binom_test(n - x, n, 1 - p)
    pnew = binomtest(n - x, n, 1 - p).pvalue
    if pold != pnew:
        print(k, x, n, p, " flipped")
