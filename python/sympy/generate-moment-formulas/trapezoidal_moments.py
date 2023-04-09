
import sympy
from sympy import exp
from genmom import generate_moments


a, b, c, d, t = sympy.symbols('a,b,c,d,t')

trapezoidal_mgf = (2/((d + c - b - a)*t**2)
                   * ((exp(d*t) - exp(c*t))/(d - c)
                      - (exp(b*t) - exp(a*t))/(b - a)))

trapezoidal_moments = generate_moments(trapezoidal_mgf, t, 4)

print()
print("Trapezoidal distribution moments about 0")
print("\nm1: ", trapezoidal_moments[0])
print("\nm2: ", trapezoidal_moments[1])
print("\nm3: ", trapezoidal_moments[2])
print("\nm4: ", trapezoidal_moments[3])
