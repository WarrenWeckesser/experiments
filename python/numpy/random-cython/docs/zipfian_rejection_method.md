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
This is what I'll call the "target" function.

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/numpy/random-cython/docs/zipfian_nnpdf.png)
