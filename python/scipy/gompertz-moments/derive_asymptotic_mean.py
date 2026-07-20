import sympy


c, t = sympy.symbols('c,t')

# Mean of the Gompertz distribution for shape parameter c,
# with loc=0 and scale=1.
f = -sympy.exp(c) * sympy.Ei(-c)

ft = f.subs(c, 1/t)
s = ft.series(n=8)
print("Asymptotic series for the mean of the Gompertz distribtion (t = 1/c)")
print(s)
