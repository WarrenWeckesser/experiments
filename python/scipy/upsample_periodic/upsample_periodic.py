
import numpy as np
from scipy.fft import fft, ifft


def upsample_periodic(x, n, dt=1, axis=-1):
    """
    Upsample a 1D real sequence by FFT interpolation.

    It is assumed that x represents samples from a periodic function
    ``x(t)``. ``dt`` is the sampling period; the period of the periodic
    function is assumed to be ``len(x)*dt``.

    The return values are ``y``and ``dt_out``.  ``y`` has length ``n``.
    ``dt_out`` is simply ``dt*len(x)/n``.

    Parameters
    ----------
    x : array_like
        Known function values of the function to be upsampled.
    n : int
        Desired number of output samples.
    dt : float, optional
        Sample period of the input function.  Default is 1.
    axis : int, optional
        Axis along which the function is given. Default is -1.

    Returns
    -------
    y : ndarray
        The upsampled function values.
    dt_out : float
        The sample period of the output values.

    Notes
    -----
    It is important to understand the sampling intervals involved in this
    method.  Suppose, for example, the function is called with

        >>> x = np.array([1.5, 1.0, -2.5, -0.5])
        >>> y, dt_out = upsample_periodic(x, 5, dt=0.5)

    The input contains four samples of a periodic function that is assumed
    to be defined on the interval [0, len(x)*dt] = [0, 2].  That is, the
    samples occur at 0, 0.5, 1.0, and 1.5::

           +--------------+--------------+--------------+--------------+
       t:  0             0.5            1.0            1.5            2.0
       x: 1.5            1.0           -2.5           -0.5

    ``upsample_periodic`` assumes that the input ``x`` is samples from a
    periodic function, so we could extend the above table to include
    ``x = 1.5`` at ``t = 2.0``.

    The value of ``dt_out`` returned by the above function call will be
    0.4.  The array ``y`` that is returned looks like::

           +-----------+-----------+-----------+-----------+-----------+
       t:  0          0.4         0.8         1.2         1.6         2.0
       y: 1.5         1.51       -1.42       -2.30        0.0831

    If we upsample to ``n=6`` instead of ``n=5`` with the function call

        >>> y, dt_out = upsample_periodic(x, 6, dt=0.5)

    then ``dt_out`` is 1/3, and the ``y`` samples look like::

           +---------+---------+---------+---------+---------+---------+
       t:  0        1/3       2/3       1.0       4/3       5/3       2.0
       y: 1.5       1.71     -0.288    -2.5      -1.59      0.413

    ``y`` will contain all the data points in ``x`` only when ``n`` is an
    integer multiple of ``len(x)``.  If ``n`` and ``len(x)`` are relatively
    prime, the only data point in ``x``` that is also in ``y`` is ``x[0]``.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> x = np.array([1.0, 2.5, -0.5, -0.25])
    >>> dt = 0.25
    >>> y, dt_out = upsample_periodic(x, 12, dt)
    >>> y
    array([ 1.        ,  1.80576905,  2.47203493,  2.5       ,  1.72203493,
            0.50673095, -0.5       , -0.86826905, -0.65953493, -0.25      ,
            0.09046507,  0.43076905])
    >>> dt_out
    0.08333333333333333

    >>> tx = np.arange(len(x)) * dt
    >>> ty = np.arange(len(y)) * dt_out
    >>> plt.plot(tx, x, 'o')
    >>> plt.plot(ty, y, '.-')
    >>> plt.grid(True)
    >>> plt.show()

    """
    x = np.asarray(x)
    x_ndim = x.ndim
    if x_ndim == 0:
        raise ValueError('x must be an array with at least one dimension.')

    x_shape = x.shape
    m = x_shape[axis]
    if m == 0:
        raise ValueError('Input x must contain at least one value.')
    if n < m:
        raise ValueError(f'n ({n}) must not be less than the number '
                         f'of samples ({m}).')

    if x_ndim > 1 and axis != 0:
        x = np.swapaxes(x, axis, 0)
    X = fft(x, axis=0)
    pad = np.zeros((n - m,) + X.shape[1:])
    b = (m + 1) // 2
    Xpad = np.r_[X[:b], pad, X[b:]]
    y = ifft(Xpad, axis=0)*(n / m).real
    if x_ndim > 0 and axis != 0:
        y = np.swapaxes(y, axis, 0)
    return y.real, dt*(m/n)
