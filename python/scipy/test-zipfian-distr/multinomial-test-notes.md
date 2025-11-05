Using the multinomial test for the Zipfian distribution
-------------------------------------------------------

Consider the concrete example a=1.25, n=3.  Using such a small n will allow
us to enumerate all the possible outcomes of the multinomial distribution
developed below.

```
In [34]: a = 1.25

In [35]: n = 3

In [36]: k = np.arange(1, n + 1)

In [37]: pmf = zipfian.pmf(k, a, n)

In [38]: pmf
Out[38]: array([0.59746908, 0.25120481, 0.15132611])
```
Some samples from this distribution:
```
In [45]: zipfian.rvs(a, n, size=15)
Out[45]: array([1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 3, 1, 2, 1, 1])
```

Consider taking 10 samples and counting the occurrences of each value
from the support.  With numpy, we can use `np.bincount`.  Because 0 is
not in the support of the distribution, the first element of the
result of `bincount(x)` will be zero, so we skip it:

```
In [68]: x = zipfian.rvs(a, n, size=10)

In [69]: y = np.bincount(x)[1:]

In [70]: y
Out[70]: array([8, 1, 1])
```

`y` is a sample from the multinomial distribution, implemented in SciPy
as `scipy.stats.multinomial`.

```
In [56]: from scipy.stats import multinomial

In [57]: multinom = multinomial(10, pmf)
```
So we can find the probability of `y`:

```
In [74]: multinom.pmf(y)
Out[74]: np.float64(0.055553039250723496)
```
In the *multinomial test* we add to that probablility the probabilities
of all other points in the support where the probability is less than or equal
to that probability to get the p-value of the test.

There are 66 points in the support of this multinomial distribution
The function `multinomial_support_generator()` was written to enumerate
all the points:

```
In [76]: supp = np.array(list(multinomial_support_generator(10, 3)))

In [77]: supp
Out[77]: 
array([[10,  0,  0],
       [ 9,  1,  0],
       [ 9,  0,  1],
       [ 8,  2,  0],
       [ 8,  1,  1],
       [ 8,  0,  2],
       [ 7,  3,  0],
       [ 7,  2,  1],
       [ 7,  1,  2],
       [ 7,  0,  3],
       [ 6,  4,  0],
       [ 6,  3,  1],
       [ 6,  2,  2],
       [ 6,  1,  3],
       [ 6,  0,  4],
       [ 5,  5,  0],
       [ 5,  4,  1],
       [ 5,  3,  2],
       [ 5,  2,  3],
       [ 5,  1,  4],
       [ 5,  0,  5],
       [ 4,  6,  0],
       [ 4,  5,  1],
       [ 4,  4,  2],
       [ 4,  3,  3],
       [ 4,  2,  4],
       [ 4,  1,  5],
       [ 4,  0,  6],
       [ 3,  7,  0],
       [ 3,  6,  1],
       [ 3,  5,  2],
       [ 3,  4,  3],
       [ 3,  3,  4],
       [ 3,  2,  5],
       [ 3,  1,  6],
       [ 3,  0,  7],
       [ 2,  8,  0],
       [ 2,  7,  1],
       [ 2,  6,  2],
       [ 2,  5,  3],
       [ 2,  4,  4],
       [ 2,  3,  5],
       [ 2,  2,  6],
       [ 2,  1,  7],
       [ 2,  0,  8],
       [ 1,  9,  0],
       [ 1,  8,  1],
       [ 1,  7,  2],
       [ 1,  6,  3],
       [ 1,  5,  4],
       [ 1,  4,  5],
       [ 1,  3,  6],
       [ 1,  2,  7],
       [ 1,  1,  8],
       [ 1,  0,  9],
       [ 0, 10,  0],
       [ 0,  9,  1],
       [ 0,  8,  2],
       [ 0,  7,  3],
       [ 0,  6,  4],
       [ 0,  5,  5],
       [ 0,  4,  6],
       [ 0,  3,  7],
       [ 0,  2,  8],
       [ 0,  1,  9],
       [ 0,  0, 10]])
```
We can use that to apply the mutinomial test to `y`.
First compute the probabilities for each point in `supp`:
```
In [78]: p = multinom.pmf(supp)
```
Then find all the values in `p` that are less than or equal to
`multinom.pmf(y)`, and add them up. That is the p-value for the
test.
```
In [80]: mask = p <= multinom.pmf(y)

In [81]: pvalue = p[mask].sum()

In [82]: pvalue
Out[82]: np.float64(0.5483587189939042)
```
So the p-value for this particular sample of 10 values from the
Zipfian distribution is approx. 0.548.

We can run the test again to generate another p-value:
```
In [94]: x = zipfian.rvs(a, n, size=10)

In [95]: y = np.bincount(x)[1:]

In [96]: y
Out[96]: array([5, 1, 4])

In [97]: mask = p <= multinom.pmf(y)

In [98]: pvalue = p[mask].sum()

In [99]: pvalue
Out[99]: np.float64(0.10317895737403479)
```

Consequences of discreteness
----------------------------

We can apply the multinomial test to many samples, and inspect
the set of p-values for any unexpected behavior.

It is important to note that, because this is a discrete distribution,
we cannot say that the p-value will be uniformly distributed on the
interval [0, 1].  That is generally true for hypothesis tests applied
to a continuous distribution, but it does not apply here.  In our
example, the support of the multinomial test has 66 elements.  This
means there are at most 66 possible p-values.  The highest possible
p-value is 1, which occurs when the element from the multinomial
distribution with the highest probability is selected.

```
In [102]: i = np.argmax(p)

In [103]: supp[i]
Out[103]: array([7, 2, 1])

In [105]: mask = p <= multinom.pmf([7, 2, 1])

In [106]: pvalue = p[mask].sum()

In [107]: pvalue
Out[107]: np.float64(1.0000000000000078)
```

With a bit more code (not shown here) we can compute all the possible
p-values for this trial. Here they are, sorted from highest to lowest
p-value:

```
[7, 2, 1]     1.00000e+00
[6, 3, 1]     9.06571e-01
[6, 2, 2]     8.14913e-01
[5, 3, 2]     7.32091e-01
[5, 4, 1]     6.62446e-01
[7, 1, 2]     6.04640e-01
[8, 1, 1]     5.48359e-01
[7, 3, 0]     4.92806e-01
[8, 2, 0]     4.41108e-01
[5, 2, 3]     3.94998e-01
[6, 4, 0]     3.53044e-01
[4, 4, 2]     3.15005e-01
[6, 1, 3]     2.78403e-01
[4, 3, 3]     2.45141e-01
[9, 1, 0]     2.15742e-01
[4, 5, 1]     1.91371e-01
[5, 5, 0]     1.67067e-01
[8, 0, 2]     1.47875e-01
[9, 0, 1]     1.31142e-01
[4, 2, 4]     1.16462e-01
[5, 1, 4]     1.03179e-01
[3, 4, 3]     9.05423e-02
[3, 5, 2]     7.81815e-02
[7, 0, 3]     6.58699e-02
[3, 3, 4]     5.45685e-02
[3, 6, 1]     4.71223e-02
[4, 6, 0]     4.03098e-02
[10, 0, 0]    3.35855e-02
[6, 0, 4]     2.77891e-02
[4, 1, 5]     2.27799e-02
[2, 5, 3]     1.95793e-02
[3, 2, 5]     1.64611e-02
[2, 6, 2]     1.37697e-02
[2, 4, 4]     1.11815e-02
[3, 7, 0]     8.83347e-03
[5, 0, 5]     7.21791e-03
[2, 7, 1]     5.69544e-03
[2, 3, 5]     4.46788e-03
[3, 1, 6]     3.33630e-03
[1, 6, 3]     2.79588e-03
[1, 5, 4]     2.35885e-03
[2, 2, 6]     1.96396e-03
[4, 0, 6]     1.62313e-03
[1, 7, 2]     1.30179e-03
[2, 8, 0]     9.90872e-04
[1, 4, 5]     7.36150e-04
[1, 8, 1]     4.98265e-04
[1, 3, 6]     3.69233e-04
[2, 1, 7]     2.73698e-04
[3, 0, 7]     2.15036e-04
[0, 6, 4]     1.68529e-04
[0, 7, 3]     1.40856e-04
[1, 2, 7]     1.14607e-04
[1, 9, 0]     8.99427e-05
[0, 5, 5]     6.61433e-05
[0, 8, 2]     4.61396e-05
[0, 4, 6]     2.97992e-05
[0, 9, 1]     1.97573e-05
[2, 0, 8]     1.37294e-05
[1, 1, 8]     9.31214e-06
[0, 3, 7]     5.59767e-06
[0, 10, 0]    2.14097e-06
[0, 2, 8]     1.14032e-06
[1, 0, 9]     3.59452e-07
[0, 1, 9]     1.10830e-07
[0, 0, 10]    6.29707e-09
```

Notice that at the top of the list (where the p-values are largest),
the spacing between the p-values is large.  A uniform distribution for
the p-value would not be a good approximation to the actual discrete
distribution of p-values.
