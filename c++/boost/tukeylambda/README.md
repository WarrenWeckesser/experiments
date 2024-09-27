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
