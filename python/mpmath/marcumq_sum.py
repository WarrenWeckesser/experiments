import mpmath
from mpmath import mp


def marcumq_addend(m, a, b):
    k = 1 - m
    while True:
        yield (a/b)**k * mp.besseli(k, a*b)
        k += 1


def marcumq_sum(m, a, b):
    with mp.extraprec(16 + 2*mp.prec):
        pre = mp.exp(-(a**2 + b**2)/2)
        s = mp.zero
        termgen = marcumq_addend(m, a, b)
        for k in range(1000):
            olds = s
            s = s + next(termgen)
            if olds == s:
                print("term too small!")
                break
        return pre * s


def _integrand(x, m, a):
    with mp.extraprec(16 + 2*mp.prec):
        e2 = mpmath.exp(-(x**2 + a**2)/2)
        return x*(x/a)**(m - 1)*e2*mpmath.besseli(m-1, a*x)


def marcumq(m, a, b):
    """
    The Marcum Q function.
    The function uses numerical integration, so it can be very slow.
    """
    with mp.extraprec(16 + 2*mp.prec):
        if a == 0:
            if m == 1:
                q = mpmath.exp(-b**2/2)
            else:
                q = mpmath.gammainc(m, b**2/2, regularized=True)
        elif b == 0 and m > 0:
            q = mpmath.mpf(1)
        else:
            q = mpmath.quad(lambda x: _integrand(x, m, a), [b, mpmath.inf])
        return q
