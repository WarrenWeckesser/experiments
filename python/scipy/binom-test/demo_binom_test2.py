
import numpy as np
from scipy.stats import binom, binom_test
import matplotlib.pyplot as plt


def _add_values(bars):
    for bar in bars:
        bb = bar.get_bbox()
        x = bb.x0 + 0.5*bb.width
        y = bb.y1
        txt = "%.3g" % y
        if txt == "1" and y < 1:
            txt = r"$<$1"
        plt.annotate(txt, (x, y), xytext=(0, 9),
                     textcoords='offset points', va='top', ha='center',
                     fontsize=8)


def demo_binom_test_plot(x, n, p):
    """
    Keep n small (i.e. less than 10 or so), otherwise the labels on the
    bars overlap.
    """
    bt_result = binom_test(x, n, p)

    k = np.arange(0, n+1)
    pmf = binom.pmf(k, n, p)

    px = pmf[x]
    include = (pmf <= px).nonzero()[0]
    exclude = (pmf > px).nonzero()[0]

    bt = pmf[include].sum()
    if not np.allclose(bt, bt_result):
        print(f"Unexpected: bt={bt}, bt_result={bt_result}")

    bars = plt.bar(include, pmf[include], color='g')
    if n <= 10:
        _add_values(bars)
    bars = plt.bar(exclude, pmf[exclude], color='k', alpha=0.4)
    if n <= 10:
        _add_values(bars)
    plt.axhline(px, linestyle='dashed', color='g', alpha=0.5)
    plt.title(f'Binomial distribution PMF, n={n}, p={p}\n'
              f'binomial test for k = {x} [sum of green bars]: {bt:8.5f}')
    plt.xlabel('k')
    if n <= 10:
        plt.xticks(k)
    plt.show()


if __name__ == "__main__":
    #x = 4
    #n = 8
    #p = 70**0.25/(70**0.25+1)
    #p = 1 / (70**0.25+1)
    x = 11
    n = 12
    p = 1 - 0.07692307692
    #x = 0
    #n = 1
    ##p = 0.5
    #p = np.nextafter(0.5, 1)
    demo_binom_test_plot(x, n, p)
