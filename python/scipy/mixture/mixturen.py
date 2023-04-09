
import numpy as np
from scipy.special import expit, logit
from scipy.stats import rv_continuous


# An attempt to create a mixture distribution as a subclass of
# scipy's rv_continuous.
#
# Check:
#   * https://github.com/scipy/scipy/pull/6800
#   * statsmodels

# There are two key design issues.  The first issue applies to any
# implementation of a mixture.  The second arises when we attempt to
# implement the mixture as a subclass of scipy.stats.rv_continuous.
#
# 1. Representation of the mixture weights.
#    The weight of each component must be between 0 and 1, and the
#    sum of the weights must be 1. Do we use the weights themselves
#    as parameters of the distribution, and deal with the constraints
#    when necessary (e.g. when doing MLE), or do we use an unconstrained
#    representation, and provide methods to convert from and to the
#    weights to that representation?
# 2. scipy.stats.rv_continuous implements a "location-scale" family.
#    rv_continous treats the location and scale parameters differently
#    than the shape parameters.  If we follow the loc-scale scheme,
#    overall location and scale parameters must be pulled out the
#    parameters of all the components. This can be done, but the result
#    is nonintuitive.  (See the implementation below.)
#    An alternative would be to, in effect, ignore the loc and scale
#    parameters of rv_continuous, and ensure all the methods of the
#    subclass are implemented to always use loc=0 and scale=1.  Then
#    the combined parameters of all the distributions become shape
#    parameters of the mixture.


# XXX The docstrings of the following conversion functions
#     are not up-to-date with what they actually do.


def _u_to_w_nd(u):
    """
    Convert the collection of u vectors in the last axis of `u`
    into mixture weights w (such that 0 <= w <= 1 and sum(w) = 1).
    """
    w = np.ones(u.shape[:-1] + (u.shape[-1] + 1,))
    u = expit(u)
    u.cumprod(axis=-1, out=w[..., 1:])
    w[..., :-1] *= 1 - u
    return w


def _w_to_u_nd(w):
    """
    Convert the collection of mixture weights in the last axis
    of `w` into the unconstrained parameters u.
    """

    # Inefficient implementation...
    # XXX This doesn't check if a denominator is 0.
    tshape = w.shape[:-1] + (w.shape[-1] - 1,)
    t = np.zeros(tshape)
    for k in range(t.shape[-1]):
        t[..., k] = ((1 - w[..., :k+1].sum(axis=-1))
                     / (1 - w[..., :k].sum(axis=-1)))
    u = logit(t)
    return u


def _u_to_w(u):
    """
    Convert the n-1 parameters u into the n mixture weights w (such that
    0 <= w <= 1 and sum(w) = 1).  Each parameter in u is an arbitrary
    real number.
    """
    # v = expit(u) maps each value in u to the interval [0, 1].
    # Then v is transformed to w.  (Note that the transformation from
    # v to w is not a bijection.)
    v = expit(u)
    w = np.ones(len(u) + 1)
    v.cumprod(out=w[1:])
    w[:-1] *= 1 - v
    return w


def _w_to_u(w):
    """
    Convert n mixture weights to n - 1 shape parameters.
    It is assumed that the values in w are nonnegative, and
    sum(w) == 1.
    """

    # Inefficient implementation...
    # XXX This doesn't check if a denominator is 0.
    t = np.zeros(len(w)-1)
    for k in range(len(t)):
        t[k] = (1 - w[:k+1].sum())/(1 - w[:k].sum())
    u = logit(t)
    return u


def _u_to_w_with_broadcasting(*args):
    """
    Convert the n-1 parameters u into the n mixture weights w (such that
    0 <= w <= 1 and sum(w) = 1).  Each parameter in u is an arbitrary
    real number.

    """
    arrs = np.array(np.broadcast_arrays(*[expit(u) for u in args]))
    arrs = np.moveaxis(arrs, 0, -1)
    w = np.ones(arrs.shape[:-1] + (len(args) + 1,))
    arrs.cumprod(axis=-1, out=w[..., 1:])
    w[..., :-1] *= 1 - arrs
    return w


class MixtureN(rv_continuous):

    def __init__(self, *dists):
        """
        Each item in dists must be a scipy continuous distribution.

        The loc and scale parameters of the first distribution become
        the loc and scale parameters of the mixture.

        The loc and scale parameters of the remaining distributions become
        shape parameters of the mixture, with names
        ``dist.name + '_loc'`` and ``dist.name + '_scale'``.

        The mixture weights for the n distributions are encoded the first
        n - 1 shape parameters.
        """
        self._dists = dists
        mix_name = '_'.join(dist.name for dist in dists) + '_mixture'

        name_count = {}
        names = []
        for dist in dists:
            name = dist.name
            count = name_count.setdefault(name, 0) + 1
            names.append(name + (count > 1)*str(count))
            name_count[name] = count

        shapes = ['u' + str(k) for k in range(1, len(dists))]
        if dists[0].shapes:
            shapes.extend([names[0] + '_' + s
                           for s in dists[0].shapes.split(',')])
        for name, dist in zip(names[1:], dists[1:]):
            if dist.shapes:
                shapes.extend([name + '_' + s for s in dist.shapes.split(',')])
            shapes.extend([name + '_loc', name + '_scale'])

        shapes = ','.join(shapes)

        a = min(dist.a for dist in dists)
        b = max(dist.b for dist in dists)
        super(MixtureN, self).__init__(a=a, b=b, name=mix_name, shapes=shapes)

    def _argcheck(self, *shapes):
        # The shape parameters of each distribution must be checked
        # by calling _argcheck() on that distribution.
        # The scale parameter of each distribution that is now a shape
        # parameter of the mixture must be greater than 0.

        nu = len(self._dists) - 1

        for k, dist in enumerate(self._dists):
            shape = shapes[nu:nu + dist.numargs]
            nu += dist.numargs
            if k > 0:
                loc, scale = shapes[nu:nu + 2]
                if np.any(scale <= 0):
                    check = False
                    break
                nu += 2
            check = dist._argcheck(*shape)
            print(check)
            if np.any(~check):
                check = False
                break
        return check

    def _pdf(self, x, *shapes):
        nu = len(self._dists) - 1
        u = shapes[:nu]
        u = np.array([z.item(0) for z in u])
        # u = np.array([float(z) for z in u])
        w = _u_to_w(u)

        p = np.zeros_like(x)
        for k, dist in enumerate(self._dists):
            shape = shapes[nu:nu + dist.numargs]
            nu += dist.numargs
            if k > 0:
                loc, scale = shapes[nu:nu + 2]
                nu += 2
            else:
                loc = 0.0
                scale = 1.0
            p += w[k]*dist.pdf(x, *shape, loc=loc, scale=scale)
        return p

    def _cdf(self, x, *shapes):
        nu = len(self._dists) - 1
        u = shapes[:nu]
        u = np.array([z.item(0) for z in u])
        w = _u_to_w(u)

        cdf = np.zeros_like(x)
        for k, dist in enumerate(self._dists):
            shape = shapes[nu:nu + dist.numargs]
            nu += dist.numargs
            if k > 0:
                loc, scale = shapes[nu:nu + 2]
                nu += 2
            else:
                loc = 0.0
                scale = 1.0
            cdf += w[k]*dist.cdf(x, *shape, loc=loc, scale=scale)
        return cdf

    def _sf(self, x, *shapes):
        nu = len(self._dists) - 1
        u = shapes[:nu]
        u = np.array([z.item(0) for z in u])
        w = _u_to_w(u)

        p = np.zeros_like(x)
        for k, dist in enumerate(self._dists):
            shape = shapes[nu:nu + dist.numargs]
            nu += dist.numargs
            if k > 0:
                loc, scale = shapes[nu:nu + 2]
                nu += 2
            else:
                loc = 0.0
                scale = 1.0
            p += w[k]*dist.sf(x, *shape, loc=loc, scale=scale)
        return p

    def _rvs(self, *params, size=(), random_state=None):
        if random_state is None:
            random_state = np.random.default_rng()
        # Scalar params only, no broadcasting, size must be () or (N,).
        if size == ():
            sz = 1
        else:
            sz = size[0]
        weights, dist_args = self.unmixed_args(*params)
        counts = random_state.multinomial(sz, weights)
        samples = []
        for dist, args, count in zip(self._dists, dist_args, counts):
            sample = dist.rvs(*args, size=count)
            samples.append(sample)
        samples = np.concatenate(samples)
        random_state.shuffle(samples)
        if size == ():
            samples = samples.item()
        return samples

    def unmixed_args(self, *params):
        """
        params must hold:
            u,
            distr1 shape parameters,
            distr2 shape parameters,
            distr2 loc, distr2 scale,
            loc,
            scale

        Returns
        -------
            w : array of float
            dist_args : tuple of tuples
        """
        nu = len(self._dists) - 1
        u = params[:nu]
        # XXX Assumes the first n-1 parameters are scalars.
        # u = np.array([z.item(0) for z in u])
        w = _u_to_w(u)

        loc, scale = params[-2:]

        dist_params = []
        for k, dist in enumerate(self._dists):
            shape = params[nu:nu + dist.numargs]
            nu += dist.numargs
            if k > 0:
                dist_loc, dist_scale = params[nu:nu + 2]
                nu += 2
                dist_loc = loc + scale * dist_loc
                dist_scale = scale * dist_scale
            else:
                dist_loc, dist_scale = loc, scale
            p = tuple(shape) + (dist_loc, dist_scale)
            dist_params.append(p)

        return w, tuple(dist_params)

    def mixture_args(self, w, *dist_args):
        u = _w_to_u(w)
        params = u.tolist()
        for k, args in enumerate(dist_args):
            if k == 0:
                dist_shape = args[:-2]
                params.extend(dist_shape)
                loc, scale = args[-2:]
            else:
                dist_shape = args[:-2]
                dist_loc, dist_scale = args[-2:]
                dist_loc = (dist_loc - loc)/scale
                dist_scale = dist_scale/scale
                params.extend(dist_shape)
                params.append(dist_loc)
                params.append(dist_scale)
        params.append(loc)
        params.append(scale)
        return params
