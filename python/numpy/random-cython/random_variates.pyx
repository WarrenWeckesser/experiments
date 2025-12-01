#cython: language_level=3

#
# This file is not being maintained at the moment.
# For the most up-to-date version of the this implementation of the rejection
# method for the finite Zipf distribution, see the SciPy pull request:
#   https://github.com/scipy/scipy/pull/24011
#

"""
Random variate generators.

This module defines the function zipfian(bitgen_t gen, a, n, size=None)
that generates variates from the Zipfian distribution (scipy.stats.zipfian)
using a rejection method.
"""

# C standard library...
from libc.math cimport log, log1p, expm1, exp, fabs, copysign, trunc
from libc.stdint cimport int64_t

# Cython...
cimport cython

# CPython...
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from cpython cimport PyFloat_AsDouble

# NumPy...
import numpy as np
cimport numpy as cnp
from numpy.random cimport bitgen_t

# SciPy Cython API from scipy.special...
from scipy.special.cython_special cimport boxcox, inv_boxcox


cnp.import_array()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Zipfian variates
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#
# validate_output_shape() copied from NumPy.
#

# XXX/TODO: Can this be made more efficient?
#           Does iter_shape have to be a Python tuple?
# Update: currently not used...
cdef validate_output_shape(iter_shape, cnp.ndarray output):
    cdef cnp.npy_intp *dims
    cdef cnp.npy_intp ndim, i
    cdef bint error
    dims = cnp.PyArray_DIMS(output)
    ndim = cnp.PyArray_NDIM(output)
    output_shape = tuple((dims[i] for i in range(ndim)))
    if iter_shape != output_shape:
        raise ValueError(
            f"Output size {output_shape} is not compatible with broadcast "
            f"dimensions of inputs {iter_shape}."
        )

#
# Functions used in the implementation of the rejection method for
# the Zipfian distribution.
#

cdef double g(double x, double a, int64_t n) noexcept nogil:
    # g is the "hat" function
    if x < 1:
        return 0
    if x < 2:
        return 1
    return (x - 1)**-a


cdef double G(double x, double a, int64_t n) noexcept nogil:
    # Currently, the only call of this function used in the
    # rejection method is G(n + 1, a, n).
    if x < 1:
        return 0.0
    if x < 2:
        return x - 1
    return boxcox(x - 1, 1 - a) + 1


cdef double Ginv(double y, double a, int64_t n) noexcept nogil:
    if 0 <= y <= 1:
        return y + 1
    return 1 + inv_boxcox(y - 1, 1 - a)


cdef double g_rv(bitgen_t *bit_generator, double a, int64_t n) noexcept nogil:
    # Generate a random variate from the dominating distribution using
    # inversion.
    # G is proportional to the CDF of the dominating distribution, but G is
    # not normalized, so to implement the inversion method, the random
    # uniform input is drawn from the interval [0, max(G)], where
    # max(G) = G(n + 1, a, n).
    cdef double g
    g = G(n + 1, a, n) * bit_generator.next_double(bit_generator.state)
    return Ginv(g, a, n)


cdef double h(double x, double a, int64_t n) noexcept nogil:
    # This is the target "histogram function" i.e. the nonnormalized PMF,
    # expanded to be a function of the continuous variable x.
    # We could return 0 for x < 1 or x > n + 1, but the function should
    # never be called with values outside that range, so let's not waste
    # time checking.
    return trunc(x)**-a


cdef int64_t zipfian_rejection(bitgen_t *bit_generator, double a, int64_t n)  noexcept nogil:
    cdef double x
    cdef int num_rejections = 0
    cdef int max_rejections = 100
    while num_rejections <= max_rejections:
        x = g_rv(bit_generator, a, n)
        # The dominating function g and the target function h coincide on the interval
        # 1 <= x < 2, so a candidate variate in that interval is never rejected.
        if x < 2 or bit_generator.next_double(bit_generator.state) * g(x, a, n) <= h(x, a, n):
            return <int64_t>(trunc(x))
        num_rejections += 1
    # Too many rejections...
    return <int64_t>(-1)


def _zipfian_value_error(varname, badvalue):
    # varname is either 'a' or 'n'.
    if varname == 'a':
        constraint = "nonnegative"
    else:
        constraint = "greater than 0"
    return ValueError(f'zipfian: {varname} must be {constraint}, got {badvalue}')


@cython.boundscheck(False)
@cython.wraparound(False)
def zipfian(bit_generator, *, a, n, size=None):
    """
    Generate random variates from the "Zipfian" distribution.
    
    The distribution is also known as the generalized Zipf distribution.
    It is a discrete distribution with finite support {1, 2, ..., n}.  The
    probability of integer k is proportional to k**-a.

    Parameters
    ----------
    bit_generator: NumPy BitGenerator instance
    a: float
        The probability of integer k in the support is proportional to k**-a.
        `a` must be nonnegative.  When `a` is 0, the distribution is uniform.
    n: int
        Determines the support {1, 2, ..., n} of the distribution.
        Must be at least 1.
    size: int or tuple of int
        Number of variates to generate.
     
    Returns
    -------
    variates: int64 or array of int64
        Zipfian random variates.
        The implementation uses a rejection method to generate the random variates.
        Theoretically, the rejection rate for this implementation should be very
        low; the average number of rejections per variate should be less than 1.
        As a precaution against extremely unlikely events (and against bugs in the
        code), the algorithm will return -1 if the number of rejections reaches 100.
 
    Examples
    --------
    >>> import numpy as np
    >>> from random_variates as zipfian

    >>> bitgen = np.random.PCG64(121263137472525314065)
    >>> zipfian(bitgen, a=1.05, n=400, size=13)
    array([  1, 374,   8,  57,   1,   1,  93,   1,  12,  17,   5,   1,   1])

    The parameters broadcast:

    >>> a = np.array([0.5, 1.25])
    >>> n = np.array([[20], [100], [500]])

    `a` has shape (2,) and `n` has shape (3, 1).  The broadcast shape is (3, 2):

    >>> zipfian(bitgen, a=a, n=n) 
    array([[  2,   1],
           [ 31,   1],
           [208,  17]])
    """
    cdef Py_ssize_t i, nvars
    cdef bitgen_t *bit_gen
    cdef const char *capsule_name = "BitGenerator"
    cdef bint is_scalar = True
    cdef cnp.ndarray variates
    cdef cnp.int64_t *variates_data
    cdef cnp.int64_t variate
    cdef cnp.broadcast it
    cdef double a1
    cdef int64_t n1
    cdef int64_t *out
    cdef variates_ndim
    cdef broadcast_ndim

    capsule = bit_generator.capsule
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("zipfian: bit_generator has an invalid capsule")
    bit_gen = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)

    try:
        a_arr = <cnp.ndarray> cnp.PyArray_FROM_OTF(a, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
    except Exception as exc:
        raise ValueError('zipfian: unable to convert the `a` argument to an '
                         'array of floating point values.') from exc
    is_scalar = is_scalar and cnp.PyArray_NDIM(a_arr) == 0

    try:
        n_arr = np.asarray(n).astype(np.int64, casting='safe', order='C')
    except Exception as exc:
        raise ValueError('zipfian: unable to convert the `n` argument to an '
                         'array of integers.') from exc
    is_scalar = is_scalar and cnp.PyArray_NDIM(n_arr) == 0

    if not is_scalar:
        if np.any(a_arr < 0):
            raise _zipfian_value_error("a", a_arr[a_arr < 0].item(0))
        if np.any(n_arr < 1):
            raise _zipfian_value_error("n", n_arr[n_arr < 1].item(0))
        if size is not None:
            variates = <cnp.ndarray> np.empty(size, np.int64)
        else:
            it = cnp.PyArray_MultiIterNew2(a_arr, n_arr)
            variates = <cnp.ndarray> np.empty(it.shape, np.int64)

        # validate_output_shape(it.shape, variates)

        # The following will raise an exception of variates, a_arr and n_arr are not
        # broadcast compatibile.
        it = cnp.PyArray_MultiIterNew3(variates, a_arr, n_arr)

        # One more validation check for the case where size is not None...
        if size is not None:
            variates_ndim = cnp.PyArray_NDIM(variates)
            broadcast_ndim = cnp.PyArray_MultiIter_NDIM(it)
            if variates_ndim != broadcast_ndim:
                raise ValueError(
                    f"The number of dimensions in the output size {np.shape(variates)} "
                    "is not equal to the number of dimensions of the broadcast shape "
                    f"{it.shape}"
                )
            for i in range(variates_ndim):
                if cnp.PyArray_MultiIter_DIMS(it)[i] != cnp.PyArray_DIMS(variates)[i]:
                    raise ValueError(
                        f"Output size {np.shape(variates)} is not compatible with broadcast "
                        f"dimensions of inputs {it.shape}."
                    )

        with bit_generator.lock, nogil:
            for i in range(cnp.PyArray_SIZE(variates)):
                a1 = (<double*> cnp.PyArray_MultiIter_DATA(it, 1))[0]
                n1 = (<int64_t*> cnp.PyArray_MultiIter_DATA(it, 2))[0]
                out = <int64_t*> cnp.PyArray_MultiIter_DATA(it, 0)
                out[0] = zipfian_rejection(bit_gen, a1, n1)
                cnp.PyArray_MultiIter_NEXT(it)

        return variates

    # a and n are scalars...

    a1 = PyFloat_AsDouble(a)
    if a1 < 0:
        raise _zipfian_value_error("a", a)
    n1 = <int64_t> n
    if n1 < 1:
        raise _zipfian_value_error("n", n)

    if size is None:
        # XXX Keep the GIL when generating just one variate.  Would it be
        # better to release it?
        with bit_generator.lock:
            variate = zipfian_rejection(bit_gen, a1, n1)
        return variate

    # a and n are scalars, size is not None...

    variates = <cnp.ndarray> np.empty(size, np.int64)
    nvars = cnp.PyArray_SIZE(variates)
    variates_data = <cnp.int64_t *> cnp.PyArray_DATA(variates)

    with bit_generator.lock, nogil:
        for i in range(nvars):
            variates_data[i] = zipfian_rejection(bit_gen, a1, n1)

    return variates
