Notes on the rejection method implementation for the Zipfian distribution
-------------------------------------------------------------------------

This is not a tutorial on the rejection method.  There are probably many of
those available already online.  These notes provide the details for the
implementation of the method for the Zipfian distribution (implemented in
SciPy as `scipy.stats.zipfian`).

The PMF for the Zipfian distribution is

$$
    f(k, a, n) = \frac{k^{-a}}{H_{n, a}}, \; k \in \{1, 2, \ldots, n\}
$$

where

$$
    H_{n, a} = \sum_{k = 1}^{n} k^{-a}
$$

is the normalization constant required to make $f$ a PMF.

For the plots, I'll use the parameters `a = 0.95` and `n = 7`.

Here is a plot of the PMF:

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/zipfian_pmf.png)

Extend this distribution to a continuous distribution with a piecewise constant PDF.
If we can generate variates from this continuous distribution, we can truncate those
variates to their integer part to get variates from the discrete Zipfian distribution.

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/zipfian_pdf.png)

Scale up by removing the normalization constant, so the value in the interval
$k \le x < k + 1$ is just $k^{-a}$ (which is easy to compute!).  This is what
I'll call the *target function* $h(x, a, n)$.

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/zipfian_nnpdf.png)

The *dominating distribution* is a distribution with a nonnormalized PDF $g(x, a, n)$ that
satisfies $g(x, a, n) \ge h(x, a, n)$ on the support.  This is the distribution that we'll
use to generate candidate random variates.  For the Zipfian distribution, we can use

$$
    g(x, a, n)
     = \begin{cases}
         1                       & 1 \le x < 2 &       \\
         \left(x - 1\right)^{-a} & 2 \le x < n + 1     \\
         0                       & \textrm{otherwise}
       \end{cases}
$$

On the inteval $2 \le x < n + 1$, the function is given by the
nonnormalized PMF formula, shifted by 1.

This plot shows the target nonnormalized PDF and the dominating nonnormalized PDF.

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/zipfian_nnpdf_and_dom.png)

To generate variates from the dominating distribution, we'll use the inversion method:
generate uniform variates and pass them through the inverse of the CDF.  To avoid
having to compute a normalization constant, we can get the nonnormalized CDF
$G(x, a, n)$ by integrating the nonnormalized PDF.

This plot shows $G(x, 0.95, 7)$:

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/zipfian_dom_nncdf.png)

To generate random variates with the inversion method from $G(x, a, n)$, we don't
have to normalize it and get a true CDF. Instead, we generate uniform variates $U$
from the interval $[0, G(n+1, a, n)]$ (since $G(n+1, a, n)$ is the value of
$G(x, a, n)$ at the right end of the support).  Then a random variate from the
dominating distribution is $X = G^{-1}(U, a, n)$.

As per the rejection method, another uniform variate $V$ is generated, this time
from $[0, 1]$, and $X$ is accepted if $V g(X, a, n) \le h(X, a, n)$.

Note that on the interval $1 \le x \lt 2$, $g(x, a, n) \equiv h(x, a, n)$.
So if the candidate $X$ is in this interval, it will always be accepted, and
there is no need to generate $V$.

When $X$ is accepted, $\lfloor X \rfloor$ is a variate from the discrete
Zipfian distribution. That is, $X$ is truncated to the largest integer not
greater than $X$ to give a variate from the Zipfian distributions.
