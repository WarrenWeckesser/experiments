Notes on the rejection method implementation for the Zipfian distribution in SciPy
----------------------------------------------------------------------------------

This is not a tutorial on the rejection method.  There are probably many of those
available already online.  These notes provide the details for the implementation
of the method in SciPy for the Zipfian distribution `scipy.stats.zipfian`.

For the plots, I'll use the parameters `a = 0.95` and `n = 7`.

Here is a plot of the PMF:

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/zipfian_pmf.png)

Extend this distribution to a continuous distribution with a piecewise constant PDF:

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/zipfian_pdf.png)

Scale up by removing the normalization constant, so the value at $x = k$ is just $k^{-a}$.
This is what I'll call the *target function* $h(x, a, n)$.

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/zipfian_nnpdf.png)

The *dominating distribution* is a distribution with a nonnormalized PDF $g(x, a, n)$ that
satisfies $g(x, a, n) \ge h(x, a, n)$ on the support.  This is the distribution that we'll
use to generate candidate random variates.  For the Zipfian distribution, we can use

$$
    g(x, a, n)
     = \begin{cases}
         1                                 & \frac{1}{2} \le x < \frac{3}{2} & \\
         \left(x - \frac{1}{2}\right)^{-a} & \frac{3}{2} \le x < n + \frac{1}{2} \\
         0                                 & \textrm{otherwise}
       \end{cases}
$$

On the inteval $\frac{3}{2} \le x < n + \frac{1}{2}$, the function is given by the PMF formula,
shifted by $\frac{1}{2}$.

This plot shows the target nonnormalized PDF and the dominating PDF.

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/zipfian_nnpdf_and_dom.png)

To generate variates from the dominating distribution, we'll use the inversion method: generate uniform
variates and pass them through the inverse of the CDF.  To avoid having to deal with a normalization
constant, we can get the nonnormalized CDF $G(x, a, n)$ by integrating the nonnormalized PDF.
This plot shows $G(x, 0.95, 7)$:

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/zipfian_dom_nncdf.png)


To generate random variates with the inversion method from $G(x, a, n)$, we don't have to normalized it
and get a true CDF. Instead, we generate uniform variates $U$ from the interval $[0, G(n+\frac{1}{2}, a, n)]$ (since
$G(n+\frac{1}{2}, a, n)$ is the value of $G(x, a, n)$ at the right end of the support).  Then a random variate
from the dominating distribution is $X = G^{-1}(U, a, n)$.

As per the rejection method, another uniform variate $V$ is generated, this time from $[0, 1]$, and $X$
is accepted if $V*g(X, a, n) \le h(X, a, n)$.

Note, however, that on the interval $1/2 \le x \lt 3/2$, $g(x, a, n) \equiv h(x, a, n)$.  So if the
candidate $X$ is in this interval, it will always be accept, and there is no need to generate $V$.
