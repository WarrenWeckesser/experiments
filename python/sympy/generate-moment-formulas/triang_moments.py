
import sympy
from sympy import exp
from genmom import generate_moments


a, b, c, t = sympy.symbols('a,b,c,t')


# Moment generating function as given in the wikipedia page
# https://en.wikipedia.org/wiki/Triangular_distribution
triang_mgf = (2*((b - c)*exp(a*t) - (b - a)*exp(c*t) + (c - a)*exp(b*t))
              / ((b - a)*(c - a)*(b - c)*t**2))

triang_moments = generate_moments(triang_mgf, t, 3)


print("Triangular distribution moments about 0")
print("m1: ", triang_moments[0])
print("m2: ", triang_moments[1])
print("m3: ", triang_moments[2])
