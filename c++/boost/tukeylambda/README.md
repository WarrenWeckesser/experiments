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
