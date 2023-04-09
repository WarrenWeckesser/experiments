
import sympy
from sympy import exp

a, b, c, d, t = sympy.symbols('a,b,c,d,t')


def generate_moments(mgf, n):
    """
    Return moments about 0 of order 1, 2, ..., n.

    The function assumes that the symbol t is the MGF parameter.
    """
    deriv = mgf
    moments = []
    for k in range(1, n+1):
        deriv = deriv.diff(t)
        moments.append(sympy.limit(deriv, t, 0))
    return moments


# Moment generating function as given in the wikipedia page
# https://en.wikipedia.org/wiki/Triangular_distribution
triang_mgf = (2*((b - c)*exp(a*t) - (b - a)*exp(c*t) + (c - a)*exp(b*t))
              / ((b - a)*(c - a)*(b - c)*t**2))

triang_moments = generate_moments(triang_mgf, 3)


print("Triangular distribution moments about 0")
print("m1: ", triang_moments[0])
print("m2: ", triang_moments[1])
print("m3: ", triang_moments[2])

trapezoidal_mgf = (2/((d + c - b - a)*t**2)
                   * ((exp(d*t) - exp(c*t))/(d - c)
                      - (exp(b*t) - exp(a*t))/(b - a)))

trapezoidal_moments = generate_moments(trapezoidal_mgf, 4)

print()
print("Trapezoidal distribution moments about 0")
print("m1: ", trapezoidal_moments[0])
print("m2: ", trapezoidal_moments[1])
print("m3: ", trapezoidal_moments[2])
print("m4: ", trapezoidal_moments[3])


k = sympy.symbols('k')
chi2_mgf = (1 - 2*t)**(-k/2)

chi2_moments = generate_moments(chi2_mgf, 3)
print()
print("chi2 distribution moments about 0")
print("m1: ", chi2_moments[0])
print("m2: ", chi2_moments[1])
print("m3: ", chi2_moments[2])
