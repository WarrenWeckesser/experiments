import numpy as np
from scipy.special import logsumexp, xlogy


# See if this is worth pursuing...

def lehmer_mean_alt(x, *, p):
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64)
    if np.any(x <= 0):
        raise ValueError('x contains 0 or negative values')

    log_num = logsumexp(xlogy(p, x))
    log_den = logsumexp(xlogy(p - 1, x))
    return np.exp(log_num - log_den)
