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
This is what I'll call the "target" function $h(x, a, n)$.

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/zipfian_nnpdf.png)

The *dominating distribution* is distribution with a nonnormalized PDF $g(x, a, n)$ that
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

This plot shows the target nonnormalized PDF and the dominating PDF.

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/zipfian_nnpdf_and_dom.png)

To generate variates from the dominating distribution, we'll use the inversion method: generate uniform
variates and pass them through the inverse of the CDF.  To avoid having to deal with a normalization
constant, we can get the nonnormalized CDF $G(x, a, n)$ by integrating the nonnormalized PDF.

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/zipfian_dom_nncdf.png)


To generate random variates with the inversion method from $G(x, a, n)$, we don't have to normalized it
and get a true CDF. Instead, we generate uniform variates $U$ from the interval $[0, G(n+1/2, a, n]$ (since
$G(n+1/2, a, n)$ is the value of $G(x, a, n)$ at the right end of the support).  Then a random variate
from the dominating distribution is $G^{-1}(U, a, n)$.
