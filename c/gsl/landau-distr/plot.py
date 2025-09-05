# This script was inspired by
# https://stackoverflow.com/questions/67003784/landau-random-number-generator
# I haven't answered, because tests that plot the PDF and a histogram of
# a large sample don't agree, and I'm not sure if the problem is in the
# levy_stable code or in the calculation of the histogram (maybe problems
# with such a heavy-tailed distribution).

import numpy as np
from scipy.stats import levy_stable
import matplotlib.pyplot as plt


def landau(mpv, scale=1):
    """
    Create a Landau distribution with its mode at mpv (the "most
    probable value") and with the given scale.

    The return value is a SciPy "frozen" distribution, with methods
    `pdf(x)`, `cdf(x)`, `rvs(size=None, random_state=None)`, etc.

    """
    # xstar was found numerically.  It is the mode of the "standard"
    # Landau distribution.
    xstar = -0.22278296
    # The Landau distribution is a special case of the Levy stable
    # family of distributions.
    return levy_stable(1, 1,
                       loc=scale*(np.log(np.pi/2) - xstar) + mpv,
                       scale=np.pi/2*scale)


x, y = np.loadtxt('out', unpack=True)

y2 = levy_stable.pdf(x, 1, 1, loc=np.log(np.pi/2), scale=np.pi/2)

plt.plot(x, y, label='Landau PDF computed with GSL', alpha=0.4, linewidth=2.5)
plt.plot(x, y2, 'k--', label='levy_stable.pdf(x, 1, 1, loc=log(pi/2), scale=pi/2',
         linewidth=1)

plt.ylim(-0.05, 0.2)
plt.legend(loc='best', framealpha=1, shadow=True)
plt.grid()
plt.show()
