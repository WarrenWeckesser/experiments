import cython

cimport numpy as cnp


ctypedef fused np_numeric_t:
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t
    cnp.float32_t
    cnp.float64_t
    cnp.longdouble_t
    cnp.complex64_t
    cnp.complex128_t

def func(const np_numeric_t[:, ::1]A):
    return 0
