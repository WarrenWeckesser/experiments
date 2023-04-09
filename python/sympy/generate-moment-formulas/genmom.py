
import sympy


def generate_moments(mgf, t, n):
    """
    Return moments about 0 of order 1, 2, ..., n.

    `t` must be the SymPy symbol that is the MGF parameter used in `mgf`.
    """
    deriv = mgf
    moments = []
    for k in range(1, n+1):
        deriv = deriv.diff(t)
        moments.append(sympy.limit(deriv, t, 0))
    return moments
