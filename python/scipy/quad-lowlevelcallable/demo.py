
import os
import ctypes
from math import exp, pow
from scipy.integrate import quad
from scipy import LowLevelCallable


# funcp is the Python implementation of the function to be integrated.
# t is the integration variable; x, a and b are additional parameters.

def funcp(t, x, a, b):
    return exp(t * x) * t**(a - 1) * (1 - t)**(b - a - 1)


lib = ctypes.CDLL(os.path.abspath('func.dylib'))
lib.func.restype = ctypes.c_double
lib.func.argtypes = (ctypes.c_int,
                     ctypes.POINTER(ctypes.c_double),
                     ctypes.c_void_p)


xab = (ctypes.c_double*3)()

x = 0.25
a = 1.0
b = 1.5
xab[0] = x
xab[1] = a
xab[2] = b
user_data = ctypes.cast(ctypes.pointer(xab), ctypes.c_void_p)
func = LowLevelCallable(lib.func, user_data)

ic = quad(func, 0, 1)
print("ic =", ic)

ip = quad(funcp, 0, 1, args=(x, a, b))
print("ip =", ip)
