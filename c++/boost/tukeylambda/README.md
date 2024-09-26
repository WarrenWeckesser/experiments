Tukey lambda distribution
=========================

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
    F(x;\lambda) \ge 1 - (-\lambda x)^{\frac{1}{\lambda}}
$$

* $\lambda < 0$, $x < 0$

$$
    F(x;\lambda) \le (-\lambda x)^{\frac{1}{\lambda}}
$$

The following plot shows the bracketing curves.

![CDF bracketing curves](https://github.com/WarrenWeckesser/experiments/blob/main/c++/boost/tukeylambda/cdf_curves.svg)
