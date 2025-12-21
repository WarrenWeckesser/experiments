#cython: language_level=3

# SciPy Cython API from scipy.special...
from scipy.special.cython_special cimport (
    agm, betaincinv, boxcox, boxcox1p, inv_boxcox, inv_boxcox1p, erfcx, jve,
    obl_ang1, voigt_profile, wright_bessel,
)

def demo():
    """
    A bunch of calls to functions from scipy.special, using the Cython API.
    """
    cdef double x = 0.5
    cdef double lam = 2.25
    cdef double y, x2
    cdef double v, a, bii, e, j, s, sp, wb

    y = boxcox(x, lam)
    x2 = inv_boxcox(y, lam)
    print("boxcox roundtip")
    print(f"{x  = }")
    print(f"{x2 = }")

    y = boxcox1p(x, lam)
    x2 = inv_boxcox1p(y, lam)
    print("boxcox1p roundtip")
    print(f"{x  = }")
    print(f"{x2 = }")
    print()

    v = voigt_profile(x, lam, 1.0)
    print(f"{v = }")

    a = agm(x, lam)
    print(f"{a = }")

    bii = betaincinv(x, lam, 0.25)
    print(f"{bii = }")

    e = erfcx(x)
    print(f"{e = }")

    j = jve(lam, x)
    print(f"{j = }")

    oa = obl_ang1(2.0, 3.0, 0.25, -0.75, &s, &sp)
    print(f"{s = }  {sp = }")

    wb = wright_bessel(1.5, lam, x)
    print(f"{wb = }")
