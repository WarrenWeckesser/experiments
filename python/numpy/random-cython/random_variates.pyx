#cython: language_level=3
"""
Random variate generators.
"""

# C standard library...
from libc.math cimport log, log1p, expm1, exp, fabs, copysign, round
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

cnp.import_array()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Zipfian variates
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#
# validate_output_shape() copied from NumPy.
#

# XXX/TODO: Can this be made more efficient?
#           Does iter_shape have to be a Python tuple?
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
# Box-Cox functions copied from SciPy.
#

cdef inline double boxcox(double x, double lmbda) noexcept nogil:
    # if lmbda << 1 and log(x) < 1.0, the lmbda*log(x) product can lose
    # precision, furthermore, expm1(x) == x for x < eps.
    # For doubles, the range of log is -744.44 to +709.78, with eps being
    # the smallest value produced.  This range means that we will have
    # abs(lmbda)*log(x) < eps whenever abs(lmbda) <= eps/-log(min double)
    # which is ~2.98e-19.  
    if fabs(lmbda) < 1e-19:
        return log(x)
    elif lmbda * log(x) < 709.78:
        return expm1(lmbda * log(x)) / lmbda
    else:
        return copysign(1., lmbda) * exp(lmbda * log(x) - log(fabs(lmbda))) - 1 / lmbda


cdef inline double inv_boxcox(double x, double lmbda) noexcept nogil:
    if lmbda == 0:
        return exp(x)
    elif lmbda * x < 1.79e308:
        return exp(log1p(lmbda * x) / lmbda)
    else:
        return exp((log(copysign(1., lmbda) * (x + 1 / lmbda)) + log(fabs(lmbda))) / lmbda)


#
# Functions using in the implementation of the rejection methods for
# the Zipfian distribution.
#

cdef double g(double x, double a, int64_t n) noexcept nogil:
    # g is the "hat" function
    if x < 0.5:
        return 0
    if x < 1.5:
        return 1
    return (x - 0.5)**-a


cdef double G(double x, double a, int64_t n) noexcept nogil:
    # Currently, the only call of this function used in the
    # rejection method is G(n + 0.5, a, n).
    if x < 0.5:
        return 0.0
    if x < 1.5:
        return x - 0.5
    return boxcox(x - 0.5, 1 - a) + 1


cdef double Ginv(double y, double a, int64_t n) noexcept nogil:
    if 0 <= y <= 1:
        return y + 0.5
    return 0.5 + inv_boxcox(y - 1, 1 - a)


cdef double g_rv(bitgen_t *bit_generator, double a, int64_t n) noexcept nogil:
    # Generate a random variate from the dominating distribution using
    # inversion.
    # G is proportional to the CDF of the dominating distribution, but G is
    # not normalized, so to implement the inversion method, the random
    # uniform input is drawn from the interval [0, max(G)], where
    # max(G) = G(n + 0.5, a, n).
    cdef double g
    g = G(n + 0.5, a, n) * bit_generator.next_double(bit_generator.state)
    return Ginv(g, a, n)


cdef double target(double x, double a, int64_t n) noexcept nogil:
    # This is the target "histogram function" i.e. the nonnormalized PMF,
    # expanded to be a function of the continuous variable x.
    # We could return 0 for x < 0.5 or x > n + 0.5, but the function should
    # never be called with values outside that range, so let's not waste
    # time checking.
    return round(x)**-a


cdef int64_t zipfian_rejection(bitgen_t *bit_generator, double a, int64_t n)  noexcept nogil:
    cdef double x
    cdef int num_rejections = 0
    cdef int max_rejections = 100
    while num_rejections <= max_rejections:
        x = g_rv(bit_generator, a, n)
        # The dominating function g and the target function coincide on the interval
        # 0.5 < x < 1.5, so a candidate variate in that interval is never rejected.
        if x <= 1.5 or bit_generator.next_double(bit_generator.state) * g(x, a, n) <= target(x, a, n):
            return <int64_t>(round(x))
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

        nvars = cnp.PyArray_SIZE(variates)

        it = cnp.PyArray_MultiIterNew3(variates, a_arr, n_arr)
        validate_output_shape(it.shape, variates)
        with bit_generator.lock, nogil:
            for i in range(nvars):
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
