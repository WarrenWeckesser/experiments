The script `curve.py` computes a curve in 3-D by integrating the Frenet-Serret
differential equations, given:

* curvature and torsion as functions of arclength
* initial point, tangent and normal vectors

These equations are

```math
\begin{align}
\frac{dT}{ds} & = \kappa(s) N \\
\frac{dN}{ds} & = -\kappa(s) T + \tau(s) B \\
\frac{dB}{ds} &= -\tau(s) N
\end{align}
```

where

* $s$ is the arclength;
* $T$, $N$ and $B$ are the tangent, normal and binormal vectors,
respecitively;
* $\kappa(s)$ is the curvature and $\tau(s)$ is the torsion.

The script creates the following plot:

![](https://github.com/WarrenWeckesser/experiments/blob/main/python/scipy/frenet/curve.svg)
