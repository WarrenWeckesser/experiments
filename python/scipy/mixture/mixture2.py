
import numpy as np
from scipy.stats import rv_continuous


# Mixture2 is an experiment that creates a discrete mixture
# of two distributions from `scipy.stats`.


class Mixture2(rv_continuous):

    def __init__(self, distr1, distr2):
        """
        distr1 and distr2 must be scipy continuous distributions.

        The loc and scale parameters of the first distribution become
        the loc and scale parameters of the mixture.

        The loc and scale parameters of the second distribution become
        shape parameters of the mixture, with names
        ``distr2.name + '_loc'`` and ``distr2.name + '_scale'``.

        """
        self._distr1 = distr1
        self._distr2 = distr2
        name = distr1.name + '_' + distr2.name + '_mixture'
        name1 = distr1.name
        name2 = distr2.name
        if name1 == name2:
            name1 += '1'
            name2 += '2'
        shapes1 = distr1.shapes.split(',') if distr1.shapes else []
        shapes2 = distr2.shapes.split(',') if distr2.shapes else []

        shapes = ','.join(['w'] +
                          [name1 + '_' + s for s in shapes1] +
                          [name2 + '_' + s for s in shapes2] +
                          [name2 + '_loc', name2 + '_scale'])
        # self.shapes = ','.join(shapes)
        super(Mixture2, self).__init__(a=min(distr1.a, distr2.a),
                                       b=max(distr1.b, distr2.b),
                                       name=name, shapes=shapes)

        # self.numargs = len(shapes)
        self._numargs1 = distr1.numargs
        self._numargs2 = distr2.numargs

    def _argcheck(self, *shapes):
        w = shapes[0]
        # shapes1 = shapes[1:1+self._numargs1]
        # shapes2 = shapes[1+self._numargs1:1+self._numargs1+self._numargs2]
        loc2, scale2 = shapes[-2:]
        if np.any((w < 0) | (w > 1)):
            return False
        if scale2 <= 0:
            return False
        # Temporarily don't check.  This should call _argcheck appropriately
        # for distr1 and distr2.
        return True

    def _pdf(self, x, *shapes):
        w = shapes[0]
        shapes1 = shapes[1:1+self._numargs1]
        shapes2 = shapes[1+self._numargs1:1+self._numargs1+self._numargs2]
        loc2, scale2 = shapes[-2:]
        p = (w*self._distr1.pdf(x, *shapes1)
             + (1 - w)*self._distr2.pdf(x, *shapes2, loc=loc2, scale=scale2))
        return p

    def _cdf(self, x, *shapes):
        w = shapes[0]
        shapes1 = shapes[1:1+self._numargs1]
        shapes2 = shapes[1+self._numargs1:1+self._numargs1+self._numargs2]
        loc2, scale2 = shapes[-2:]
        p = (w*self._distr1.cdf(x, *shapes1)
             + (1 - w)*self._distr2.cdf(x, *shapes2, loc=loc2, scale=scale2))
        return p

    def _sf(self, x, *shapes):
        w = shapes[0]
        shapes1 = shapes[1:1+self._numargs1]
        shapes2 = shapes[1+self._numargs1:1+self._numargs1+self._numargs2]
        loc2, scale2 = shapes[-2:]
        p = (w*self._distr1.sf(x, *shapes1)
             + (1 - w)*self._distr2.sf(x, *shapes2, loc=loc2, scale=scale2))
        return p

    def unmixed_args(self, *params):
        """
        params must hold:
            w,
            distr1 shape parameters,
            distr2 shape parameters,
            distr2 loc, distr2 scale,
            loc,
            scale

        Returns
        -------
            w : float
            distr1_params : tuple
                *distr1_shapes, loc, scale
            distr2_params : tuple
                *distr1_shapes, loc2, scale2
        """
        w = params[0]
        shapes1 = params[1:1+self._numargs1]
        shapes2 = params[1+self._numargs1:1+self._numargs1+self._numargs2]
        loc2, scale2 = params[-4:-2]
        loc, scale = params[-2:]
        loc2 = loc + scale*loc2
        scale2 = scale * scale2

        return (w, tuple(shapes1) + (loc, scale),
                tuple(shapes2) + (loc2, scale2))
