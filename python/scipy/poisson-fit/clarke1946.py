"""
This is the calculation given in the paper

    R. D. Clarke, An application of the Poisson distribution, Journal of the
    Institute of Actuaries, vol. 72 (1946), p. 48.

It is given as an example Feller on page 150 of "An Introduction to Probability
Theory And Its Applications (vol 1)", 2nd ed.

The data in Feller and in the original article is presented like this:

     Flying-bomb hits on London
  k       0     1    2    3    4    5 and over     
  Nk    229   211   93   35    7    1

and we are told that the total number of bombs was 537.
Each Nk counts how many small regions of a grid of 576 square kilometer
regions were hit with k bombs.

It is important to note that the heading of the last column is "5 and over",
and that only one region was hit with 5 or more bombs.  If we simply
compute the total number regions as 229*0 + 211*1 + 93*2 + 35*3 +7*4 + 1*5, 
we get 535, but we are told that 537 bombs hit the area.  To get the total
to be 537, the actual number of bombs that hit the region with "5 and over"
hits must be 7.  So the complete table is

  k       0     1    2    3    4    5    6   7
  Nk    229   211   93   35    7    0    0   1

This is the data that is used in this script.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Regarding the Χ² test, Clarke (1946) says:

    Applying the Χ² test to the comparison of actual with expected
    figures, we obtain Χ²=1.17.  There are 4 degrees of freedom, and
    the probability of obtaining this or a higher value of Χ² is .88.

Why are there 4 degrees of freedom?  The parameter mu is estimated
from the data, and that determines the only the probability distribution.
The expected values use N, also derived from the data.

In general, when determining the degrees of freedom for a chi-square
test:
* If the observed frequencies are not used *at all* to determine the
  expected frequencies (i.e. the expected frequencies are given),
  then dof = len(observed).
* If the probability distribution of the expected frequencies is given
  (i.e. the distribution is fixed), and the actual expected frequencies
  are computed by multiplying the PMF by the total of the observed
  frequencies, then dof = len(observed) - 1.
* If the probability distribution has one parameter, and that parameter
  is estimated from the observed frequencies, and then the expected
  frequencies are computed as above, then dof = len(observed) - 2.

"""
import numpy as np
from scipy.stats import poisson, chisquare


hits = np.array([229, 211, 93, 35, 7, 0, 0, 1])

k = np.arange(len(hits))

total = hits.dot(k)
print("total hits =", total)
N = hits.sum()
print("N =", N)

mu = total / N
print("mu = %9.7f" % mu, "  (MLE estimate of the Poisson parameter)")
print()

expected = poisson.pmf(k, mu) * N

print("       k  ", end='')
for i in k:
    print("%3d    " % i, end=' ')
print()
print("-"*75)
print("  actual  ", end='')
for i in k:
    print("%3d    " % hits[i], end=' ')
print()
print("expected  ", end='')
for i in k:
    print("%7.3f" % expected[i], end=' ')
print()
print(" "*50, '+---------------------+')
print(" "*57, '%7.3f (5 and over)' % expected[-3:].sum())

# Observed frequencies, with k = 5, 6, 7 binned.
f_obs = np.concatenate((hits[:-3], [hits[-3:].sum()]))

# Expected frequencies, with k = 5, 6, 7 binned.
f_exp = np.concatenate((expected[:-3], [expected[-3:].sum()]))

# According to Clarke (1946), dof = 4.
# The default dof used by chisquare is len(obs) - 1, which would
# be 5 in this example.  We want dof = 4, so we'll use the argument
# ddof=1, which substracts one from the default dof.
stat, p = chisquare(f_obs, f_exp, ddof=1)
print("Χ² = %5.3f" % stat)
print("p = %6.4f" % p)


