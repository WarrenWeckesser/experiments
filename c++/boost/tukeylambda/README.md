Tukey lambda distribution
-------------------------

The quantile function (i.e. the inverse of the CDF) is

$$
Q(p;\lambda)= 
  \begin{cases}
    \frac{1}{\lambda}\left(p^{\lambda} - (1 - p)^{\lambda}\right),  & \textrm{if } \lambda \neq 0\\
    \log\left(\frac{p}{1 - p}\right),                               & \textrm{if } \lambda = 0
  \end{cases}
$$

The derivative of $Q$ is

$$
q(p;\lambda) = \frac{dQ}{dp} = p^{\lambda - 1} + (1 - p)^{\lambda - 1}
$$

Numerical issues
----------------
* $Q(p;\lambda)$ must be inverted numerically to find the CDF $F(x; \lambda)$.
  The current implementation in SciPy needs improvement.
* The straightforward implementation of $Q(p;\lambda)$ in code loses precision
  for some parameter ranges.  Precision is lost when the terms in the subtraction
  are close.  This occurs when $p \approx 0.5$ and/or when $|\lambda|$ is small.
  Loss of precision in $Q(p;\lambda)$ limits the precision that can be expected
  in the numerical inversion of $Q(p;\lambda)$.


Inverting $Q(p;\lambda)$
------------------------

The following bounds can be derived; these are useful for bracketing the search
when computing the CDF numerically.

* $\lambda < 0$, $x > 0$

$$
    F(x;\lambda) > 1 - (-\lambda x)^{\frac{1}{\lambda}}
$$

* $\lambda < 0$, $x < 0$

$$
    F(x;\lambda) < (\lambda x)^{\frac{1}{\lambda}}
$$

The following plot shows the bracketing curves.  The brackets shown in the
plot are implemented in the function `get_cdf_solver_bracket()` in `tukeylambda.h`.

![CDF bracketing curves](https://github.com/WarrenWeckesser/experiments/blob/main/c++/boost/tukeylambda/cdf_curves.svg)

(The red dotted line could also be used to refine the bracket, but it is
not implemented in the C code.  I haven't tested whether the slightly more
expensive set up and computation of the bracket would be offset by what
would probably be just one less iteraton of the numerical solver.)

*Derivation of the bracketing curves*

If $x$ is far into the left tail (i.e. $x < 0$ and $|x|$ is "big"), then $p$ is "small",
and in $Q(p;\lambda)$, the term $p^{\lambda}$ will be much larger than $(1 - p)^{\lambda}$ (because $\lambda < 0$).  To derive an approximate inverse of $Q$ in this case, we ignore
the term $(1 - p)^{\lambda}$ and invert

$$
  \hat{Q}(p, \lambda) = \frac{p^{\lambda}}{\lambda}
$$

Solving $x = \hat{Q}(p;\lambda)$ for $p$ gives

$$
    p = (\lambda x)^{\frac{1}{\lambda}}
$$

That is upper curve of the bracket for $x < \frac{2^{-\lambda}}{\lambda}$.

Similarly, if $x$ is far into the right tail ($x > 0$ and $x$ is "big"), then $p$
is close to $1$, and $1 - p$ is "small". In this case, the term $(1 - p)^{\lambda}$
will be much larger than $p^{\lambda}$, and the approximation $\hat{Q}$ is

$$
  \hat{Q}(p, \lambda) = -\frac{(1 - p)^{\lambda}}{\lambda}
$$

Solving $x = \hat{Q}(p;\lambda)$ for $p$ gives

$$
    p = 1 - (-\lambda x)^{\frac{1}{\lambda}}
$$

That is the lower curve of the bracket for $x > \frac{-2^{-\lambda}}{\lambda}$.


Loss of precision in $Q(p; \lambda)$ when $\lambda$ is small
------------------------------------------------------------

The straightforward implementation of $Q(p; \lambda)$ suffers
from loss of precision when $\lambda$ is very small, and when $p$ is
close to $\frac{1}{2}$.

For example,

```
In [47]: from mpsci.distributions import tukeylambda

In [48]: from mpmath import mp

In [49]: mp.dps = 400

In [50]: def quantile(p, lam):
    ...:     # Simple implementation
    ...:     return (p**lam - (1 - p)**lam)/lam
    ...: 

In [51]: p = 0.500005

In [52]: lam = 1e-10

In [53]: quantile(p, lam)
Out[53]: 1.9984014443252818e-05

In [54]: float(tukeylambda.invcdf(p, lam))
Out[54]: 1.9999999999411395e-05
```

See https://github.com/scipy/scipy/issues/21370 for more discussion
and examples.

*Recurrence relation for* $Q(p;\lambda)$

Here's an interesting approach that seems to work pretty well,
but might be too slow to be worthwhile.  It would probably be simpler
and faster to just switch to double-double precision in the region
where the loss of precision in the simple implementation is
unacceptable.

The subtraction in $Q(p;\lambda)$ has the form $a^{\lambda} - b^{\lambda}$.
We use the "difference of powers" formula to rewrite this expression as follows:

$$
\begin{split}
a^{\lambda} - b^{\lambda}
  & = \left(a^{\lambda/2}\right)^2 - \left(b^{\lambda/2}\right)^2 \\
  & = \left(a^{\lambda/2} - b^{\lambda/2}\right)\left(a^{\lambda/2} + b^{\lambda/2}\right)
\end{split}
$$

When this is applied to $Q(p;\lambda)$, we have

$$
\begin{split}
Q(p;\lambda)
  & = \frac{1}{\lambda}\left(p^{\lambda} - (1 - p)^{\lambda}\right) \\
  & = \frac{\left(p^{\lambda/2} - (1 - p)^{\lambda/2}\right)}{\lambda/2}
      \frac{\left(p^{\lambda/2} + (1 - p)^{\lambda/2}\right)}{2} \\
  & = Q(p; \lambda/2) \frac{\left(p^{\lambda/2} + (1 - p)^{\lambda/2}\right)}{2}
\end{split}
$$

If we apply the algebraic change to $Q(p; \lambda/2)$ we obtain

$$
\begin{split}
Q(p;\lambda)
  & = Q(p, \lambda/2) \frac{\left(p^{\lambda/2} + (1 - p)^{\lambda/2}\right)}{2} \\
  & = Q(p, \lambda/4) \frac{\left(p^{\lambda/4} + (1 - p)^{\lambda/4}\right)}{2}
                      \frac{\left(p^{\lambda/2} + (1 - p)^{\lambda/2}\right)}{2}
\end{split}
$$

Repeating the process $n$ times gives

$$
Q(p;\lambda)
  = Q\left(p, \frac{\lambda}{2^n}\right) \prod_{k=1}^n\frac{1}{2}\left(p^{\lambda/2^k} + (1 - p)^{\lambda/2^k}\right)
$$

So far, this is an exact expression. When $\frac{\lambda}{2^n}$ is sufficiently small,
$Q(p;\lambda/2^n)$ can be approximated with $\log\left(\frac{p}{1 - p}\right)$,
giving

$$
Q(p;\lambda) \approx 
   \log\left(\frac{p}{1-p}\right)
   \prod_{k=1}^n\frac{1}{2}\left(p^{\lambda/2^k} + (1 - p)^{\lambda/2^k}\right)
$$

This idea is implemented in the Python function `tl_invcdf(p, lam, n=None)`
in the file `tl_invcdf_small_lambda.py`, and in the C++ file `tukeylambda.h` as
the function `tukey_lambda_invcdf_experimental(p, lam)`.
Here it is applied to the previous example:

```
In [13]: tl_invcdf(p, lam, 20)[0].item()
Out[13]: 1.999999999941139e-05
```

With $n=20$, it computes the result to within one ULP of the exact result.
For moderate to large values of $\lambda$, it is typically necessary for $n$
to be around 55 or so to get results that are close to machine precision.

So it works, but more testing and development is needed to see if
it could compete with just switching to double-double precision or using
some of the ideas discussed in https://github.com/scipy/scipy/issues/21370.

*Alternative formulations of* $Q(p;\lambda)$

$$
\begin{split}
Q(p;\lambda)
  & = \frac{1}{\lambda}\left(p^{\lambda} - (1 - p)^{\lambda}\right) \\
  & = \frac{p^{\lambda}}{\lambda}\left(1 - \left(\frac{1-p}{p}\right)^{\lambda}\right) \\
  & = \frac{p^{\lambda}}{\lambda}\left(1 - e^{\lambda \log\left(\frac{1-p}{p}\right)}\right) \\
  & = \frac{p^{\lambda}}{\lambda}\left(1 - e^{-\lambda \textrm{logit}(p)}\right) \\
  & = -\frac{p^{\lambda}}{\lambda}\textrm{expm1}(-\lambda \textrm{logit}(p))
\end{split}
$$


*Taylor series in* $\lambda$

Expand $Q(p;\lambda)$ in a Taylor series about $\lambda = 0$ to obtain

$$
\begin{split}
Q(p;\lambda)
    &= \sum_{k = 1}^{\infty} \frac{\lambda^{k-1}}{k!}\left(\log^{k}(p) - \log^{k}(1 - p)\right) \\
    &= \left(\log(p) - \log(1 - p)\right)
        \sum_{k=0}^{\infty}\left(
                             \frac{\lambda^{k}}
                                  {(k+1)!}
                             \sum_{j=0}^{k}\log^{k-j}(p)\log^{j}(1 - p)
                           \right)
\end{split}
$$

The second equality uses the "difference of powers" formula.
The inner sum is over all the $k$-th order binomial powers of $\log(p)$ and $\log(1 - p)$.

This is implemented in the C++ file `tukeylambda.h` as
the function `tukey_lambda_invcdf_taylor(p, lam, n)`.

For the example above, where $p$ is `0.500005` and $\lambda$ is `1e-10`, and with
`n = 3`, the function computes the result to full machine precision.