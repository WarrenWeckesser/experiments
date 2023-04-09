
import sympy
from genmom import generate_moments


k, t = sympy.symbols('k,t')

chi2_mgf = (1 - 2*t)**(-k/2)

chi2_moments = generate_moments(chi2_mgf, t, 4)
print()
print("chi2 distribution moments about 0")
print("m1: ", chi2_moments[0])
print("m2: ", chi2_moments[1])
print("m3: ", chi2_moments[2])
print("m4: ", chi2_moments[3])
