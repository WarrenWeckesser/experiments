# Scratch work for timing https://github.com/scipy/scipy/pull/25696

import timeit
import numpy as np
from scipy.interpolate import AAA


rng = np.random.default_rng(121263137472525314065)

# Sizes of the inputs of the constructor AAA()
interp_data_sizes = [10, 100, 1000, 10000]
# Sizes of the inputs to the __call__ method of AAA()
eval_sizes = [10, 100, 1000, 10000]

print("n is the size of the data that used to create the interpolator.")
print("m is the size of the input evaluated by the interpolator.")
print()
print("          |                      m")
print("       n  |", end='')
for m in eval_sizes:
    print(f"{m:12}", end='')
print()
print('-'*(13 + 12*len(eval_sizes)))
for n in interp_data_sizes:
    x = rng.standard_normal(size=(n, 2)).view(np.complex128)[:, 0]
    fx = np.sin(x)
    r = AAA(x, fx)
    print(f"{n:8}  |", end='')
    for m in eval_sizes:
        z = rng.standard_normal(size=(m, 2)).view(np.complex128)[:, 0]
        code = "r(z)"
        t = timeit.timeit(code, globals=globals(), number=5000)
        print(f"{t:12.5f}", end='')
    print()
