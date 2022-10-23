
import numpy as np
from scipy.special import wofz


_SQRTPI = np.sqrt(np.pi)
_SQRTPI2 = _SQRTPI/2


def voigtuv(x, t):
    """
    Voigt functions U(x,t) and V(x,t).

    The return value is U(x,t) + 1j*V(x,t).
    """
    sqrtt = np.sqrt(t)
    z = (1j + x)/(2*sqrtt)
    w = wofz(z) * _SQRTPI2 / sqrtt
    return w


def voigth(a, u):
    """
    Voigt function H(a, u).
    """
    x = u/a
    t = 1/(4*a**2)
    voigtU = voigtuv(x, t).real
    h = voigtU/(a*_SQRTPI)
    return h


if __name__ == "__main__":
    # Create plots of U(x,t) and V(x,t) like those
    # in http://dlmf.nist.gov/7.19

    import matplotlib.pyplot as plt

    xmax = 15
    x = np.linspace(0, xmax, 100)
    t = np.array([0.1, 2.5, 5, 10])
    y = voigtuv(x, t.reshape(-1, 1))

    colors = ['g', 'r', 'b', 'y']
    styles = ['-', '--', '-.', ':']

    plt.subplot(2, 1, 1)
    for tval, ya, clr, sty in zip(t, y, colors, styles):
        plt.plot(x, ya.real, clr, linestyle=sty, label=f"t={tval:4.1f}")
    plt.grid()
    plt.xlim(0, xmax)
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.title('Voigt U(x,t)')

    plt.subplot(2, 1, 2)
    for tval, ya, clr, sty in zip(t, y, colors, styles):
        plt.plot(x, ya.imag, clr, linestyle=sty, label=f"t={tval:4.1f}")
    plt.grid()
    plt.xlim(0, xmax)
    plt.ylim(0, 0.5)
    plt.legend(loc='best')
    plt.title('Voigt V(x,t)')
    plt.xlabel('x')

    plt.tight_layout()

    plt.savefig('images/voigt.svg')

    # plt.show()
