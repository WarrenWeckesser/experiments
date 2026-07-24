import sympy as sp

eta = sp.symbols('eta', positive=True)

expr = (
    sp.exp(eta) * (
        -2*eta*sp.hyper([1,1,1],[2,2,2], -eta)
        + (sp.EulerGamma + sp.log(eta))**2
        + sp.pi**2/6
    )
    - (sp.exp(eta)*sp.E1(eta))**2
)

result1 = sp.series(sp.expand(expr), eta, 0, 1)
result2 = sp.series(sp.expand(expr), eta, 0, 2)
result3 = sp.series(sp.expand(expr), eta, 0, 3)

print(result1)
print()
print(result2 - result1.removeO())
print()
print(result3 - result2.removeO())
