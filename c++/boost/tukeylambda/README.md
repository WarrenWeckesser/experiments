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

$Q(p;\lambda)$ must be inverted numerically to find the CDF $F(x; \lambda)$.

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

The straigtforward implementation of $Q(p; \lambda)$ suffers
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

Here's an interesting approach the seems to work pretty well,
but is likely too slow to be worthwhile.  It would probably be simpler
and faster to just switch to double-double precision in the region
where the lost of precision is nontrivial.

The subtraction in $Q(p;\lambda)$ has the form $a^{\lambda} - b^{\lambda}$.
We do a little algebraic trick to rewrite this expression as follows:

$$
\begin{align*}
a^{\lambda} - b^{\lambda}
  & = \left(a^{\lambda/2}\right)^2 - \left(b^{\lambda/2}\right)^2 \\
  & = \left(a^{\lambda/2} - b^{\lambda/2}\right)\left(a^{\lambda/2} + b^{\lambda/2}\right)
\end{align*}
$$

When this is applied to $Q(p;\lambda)$, we have

$$
\begin{align*}
Q(p;\lambda)
  & =  \frac{1}{\lambda}\left(p^{\lambda} - (1 - p)^{\lambda}\right) \\
  & = \frac{\left(p^{\lambda/2} - (1 - p)^{\lambda/2}\right)}{\lambda/2}
      \frac{\left(p^{\lambda/2} + (1 - p)^{\lambda/2}\right)}{2} \\
  & = Q(p, \lambda/2) \frac{\left(p^{\lambda/2} + (1 - p)^{\lambda/2}\right)}{2}
\end{align*}
$$

If we apply the algebraic change to $Q(p; \lambda/2)$ we obtain

$$
\begin{align*}
Q(p;\lambda)
  & = Q(p, \lambda/2) \frac{\left(p^{\lambda/2} + (1 - p)^{\lambda/2}\right)}{2} \\
  & = Q(p, \lambda/4) \frac{\left(p^{\lambda/4} + (1 - p)^{\lambda/4}\right)}{2}
                      \frac{\left(p^{\lambda/2} + (1 - p)^{\lambda/2}\right)}{2}
\end{align*}
$$

Repeating the process $n$ times gives

$$
\begin{align*}
Q(p;\lambda)
  & = Q(p, \frac{\lambda}{2^n}) \prod_k^n\frac{\left(p^{\lambda/2^k} + (1 - p)^{\lambda/2^k}\right)}{2}
\end{align*}
$$

So far, this is an exact expression. When $\frac{\lambda}{2^n}$ is sufficiently small,
$Q(p;\lambda/2^n)$ can be approximated with $\log\left(\frac{p}{1 - p}\right)$,
giving

$$
Q(p;\lambda) \approx 
   \log\left(\frac{p}{1-p}\right)
   \prod_k^n\frac{\left(p^{\lambda/2^k} + (1 - p)^{\lambda/2^k}\right)}{2}
$$
