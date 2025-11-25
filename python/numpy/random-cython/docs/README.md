Notes on the implementation of the rejection method for the Zipfian distribution
--------------------------------------------------------------------------------

This is not a tutorial on the rejection method.  The method is described in the
wikipedia article [Rejection sampling](https://en.wikipedia.org/wiki/Rejection_sampling),
and a web search should point to at least a few tutorials.

These notes provide the details for the implementation of the method for the
Zipfian distribution (implemented in SciPy as `scipy.stats.zipfian`).

The PMF for the Zipfian distribution is

$$
    f(k, a, n) = \frac{k^{-a}}{H_{n, a}}, \quad k \in \\{1, 2, ..., n\\}
$$

where the *generalized harmonic number*

$$
    H_{n, a} = \sum_{k = 1}^{n} k^{-a}
$$

is the normalization constant required to make $f$ a PMF.

For the plots, I'll use the parameters `a = 0.95` and `n = 7`.

Here is a plot of the PMF:

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/zipfian_pmf.png)

Extend this distribution to a continuous distribution with a piecewise constant PDF
on support $1 \le x \lt n + 1$.  If we can generate variates from this continuous
distribution, we can truncate those variates to their integer part to get variates
from the discrete Zipfian distribution.

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/zipfian_pdf.png)

Scale up by removing the normalization constant, so the value in the interval
$k \le x < k + 1$ is just $k^{-a}$ (which is easy to compute!).  This is what
I'll call the *target function* $h(x, a, n)$.

$$
    h(x, a, n) =
        \begin{cases}
          \lfloor x \rfloor^{-a}  & 1 \le x < n + 1 &       \\
          0                       & \textrm{otherwise}
        \end{cases}
$$

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/zipfian_nnpdf.png)

The *dominating distribution* is a distribution with a nonnormalized PDF $g(x, a, n)$ that
satisfies $g(x, a, n) \ge h(x, a, n)$ on the support. (In the wikipedia article linked above,
this is called the *proposal distribution*.)  This is the distribution that we'll
use to generate candidate random variates.  For the Zipfian distribution, we can use

$$
    g(x, a, n)
     = \begin{cases}
         1                       & 1 \le x < 2 &       \\
         \left(x - 1\right)^{-a} & 2 \le x < n + 1     \\
         0                       & \textrm{otherwise}
       \end{cases}
$$

On the interval $2 \le x < n + 1$, the function is given by the
nonnormalized PMF formula, shifted by 1.

This plot shows the target nonnormalized PDF and the dominating nonnormalized PDF.

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/zipfian_nnpdf_and_dom.png)

To generate variates from the dominating distribution, we'll use the inversion method:
generate uniform variates and pass them through the inverse of the CDF.  We get the
nonnormalized CDF $G(x, a, n)$ by integrating the nonnormalized PDF.  On the interval
$1 \le x \le n + 1$, we have

$$
    G(x, a, n)
     = \begin{cases}
         x - 1                                      & 1 \le x < 2 &       \\
         \frac{\left(x - 1\right)^{1-a} - a}{1 - a} & 2 \le x < n + 1
       \end{cases}
$$

which can be written in terms of the
[Box-Cox power transformation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html)
$B(x, \lambda)$ as

$$
    G(x, a, n)
     = \begin{cases}
         x - 1                & 1 \le x < 2 &     \\
         B(x - 1, 1 - a) + 1  & 2 \le x < n + 1
       \end{cases}
$$

This plot shows $G(x, 0.95, 7)$:

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/zipfian_dom_nncdf.png)

To generate random variates with the inversion method from $G(x, a, n)$, we don't
have to normalize it and get a true CDF. Instead, we generate uniform variates $Y$
from the interval $[0, G(n+1, a, n)]$ (since $G(n+1, a, n)$ is the value of
$G(x, a, n)$ at the right end of the support).  Then a random variate from the
dominating distribution is $X = G^{-1}(Y, a, n)$.  On the interval $0 \le y \le G(n+1, a, n)$,
$G^{-1}$ is

$$
    G^{-1}(y, a, n) =
        \begin{cases}
            y + 1                     & 0 \le x < 1 &     \\
            B^{-1}(y - 1, 1 - a) + 1  & 1 \le y < G(n+1, a, n)
       \end{cases}
$$

As per the rejection method, another uniform variate $U$ is generated, this time
from $[0, 1]$, and $X$ is accepted if $U g(X, a, n) \le h(X, a, n)$.

Note that on the interval $1 \le x < 2$, $g(x, a, n) \equiv h(x, a, n)$.
So if the candidate $X$ is in this interval, it will always be accepted, and
there is no need to generate $U$.

When $X$ is accepted, $\lfloor X \rfloor$ is a variate from the discrete
Zipfian distribution. That is, $X$ is truncated to the largest integer not
greater than $X$ to give a variate from the Zipfian distributions.

Putting it all together, we have the following Python-ish pseudocode for
the Zipfian rejection method; `uniform(a, b)` generates a sample from the
uniform distribution $U(a, b)$:

```
def dominating_random_variate(a, n):
    # Use the inversion method...
    y = uniform(0, G(n + 1, a, n))
    return Ginv(y, a, n)

def zipfian_random_variate(a, n):
    # Rejection method...
    while True:
        # Generate a candidate.
        x = dominating_random_variate(a, n)
        if x < 2:
            # Always accept x in this range. (By construction, we know x >= 1.)
            return floor(x)
        U = uniform(0, 1)
        if U*g(x, a, n) <= h(x, a, n):
            return floor(X)
```

-----

With a rejection method, there is a constant (called $M$, for example, in the
wikipedia article) that is used in determining whether or not to accept a candidate
variate, and that determines the average number of iterations required by the method.
It appears to be missing in the above discussion.

The constant is "missing" because the nonnormalized PDFs are used, and the
constant that one would usually see in a rejection method has been, in effect,
absorbed into the nonnormalized PDFs.  It can be derived to be

$$
    M(a, n) = \frac{B(n, 1 - a) + 1}{H_{n, a}}
$$

That is, it is the ratio of the normalizing constants.

Numerical experiments show that $M(a, n) < 1.25$.  For each $n$, the maximum
of $M(a, n)$ occurs near $a = 3$, as seen in the following plot:

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/rejection_scaling_constant_plots.png)
