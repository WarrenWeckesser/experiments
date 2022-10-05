The file ``upsample_periodic.py`` defines a function that upsamples
a data set of evenly spaces samples, under the assumption that the
samples are from a continuous periodic function.

The following plot is generated by the script ``upsample_periodic_example.py``.
In this example, we assume that the data, ``[2.25, 1.0, -1.5, -1.25, 0.25, 0.25]``
are samples from a function ``x(t)`` that is periodic with period T = 3.
The sample spacing Δt = 0.5; i.e. the sampling frequency is 2 samples per time
unit.  (Note that the data set does not repeat the redundant sample at t=3,
nor do the graphs in the plot below.)

![](https://github.com/WarrenWeckesser/experiments/blob/master/python/scipy/upsample_periodic/upsample_periodic_example.svg)