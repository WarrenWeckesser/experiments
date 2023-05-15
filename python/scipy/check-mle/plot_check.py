import warnings
import numpy as np
from scipy.optimize import minimize, OptimizeWarning
import matplotlib.pyplot as plt


_nelder_mead_options = dict(disp=False,
                            xatol=1e-9,
                            fatol=1e-12,
                            maxiter=100000,
                            maxfev=100000)


def optimizer(func, p0, args=None, disp=0):
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=OptimizeWarning)
        result = minimize(func, p0, args=args,
                          method='nelder-mead',
                          options=_nelder_mead_options)
    if not result.success:
        raise RuntimeError(f'minimize failed; {result.status=}  '
                           f'{result.message=}')
    return result.x


def check_mle(dist, x, fixed, gridsize=500):
    """
    Fit the distribution `dist` to the data in `x`, and plot cross-sections
    of the negative log-likelihood function for the parameter values that
    were not fixed.

    `x` can be a 1-d array or an instance of `scipy.stats.CensoredData`.
    """
    x = np.asarray(x)
    opt_params = dist.fit(x, **fixed, optimizer=optimizer)
    shape_names = dist.shapes.replace(' ', '').split(',') if dist.shapes else []
    names = shape_names + ['loc', 'scale']
    for k, (name, pval) in enumerate(zip(names, opt_params)):
        print(f'{k:4d}: {name:>10s} = {pval}')
        if 'f' + name in fixed:
            continue
        # XXX This choice of interval to plot doesn't work if pval is 0.0.
        bounds = [0.975*pval, 1.025*pval]
        pgrid = np.linspace(min(bounds), max(bounds), gridsize)
        params = list(opt_params)
        pvals = []
        nll = []
        for psample in pgrid:
            params[k] = psample
            try:
                # Use the private method because it can handle CensoredData
                # instances.
                v = dist._penalized_nnlf(params, x)
            except Exception:
                v = np.nan
            if np.isfinite(v):
                pvals.append(psample)
                nll.append(v)
        fig, ax = plt.subplots()
        ax.plot(pvals, nll)
        params[k] = pval
        ax.plot(pval, dist._penalized_nnlf(params, x), 'k.',
                label=f'MLE: {name}={pval}')
        ax.set_xlabel(name)
        ax.set_ylabel('negative log-likelihood')
        ax.grid(True, alpha=0.75)
        ax.legend(framealpha=1, shadow=True)
    plt.show()
