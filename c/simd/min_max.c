// XXX No one should use this code as an example of
//     *good* SIMD programming!

#include <math.h>
#include <immintrin.h>


//
// For these min and max functions,  n must be greater than 0.
// The function does not check this.
//
// The functions float_{OP}(), float_{OP}_sse() and float_{OP}_avx
// are  NAN-oblivious: they do not take into account that an input
// value could be NAN, so the results for an input with NAN
// are undefined.
//

#define FLOAT_REDUCE_INFIX_BINOP(name, binop) \
float float_ ## name ## _keep_nan(size_t n, float *x)           \
{                                                               \
    float arg = x[0];                                           \
    if (isnan(arg)) {                                           \
        return NAN;                                             \
    }                                                           \
    for (size_t i = 1; i < n; ++i) {                            \
        if (isnan(x[i])) {                                      \
            return NAN;                                         \
        }                                                       \
        arg = (arg binop x[i]) ? arg : x[i];                    \
    }                                                           \
    return arg;                                                 \
}                                                               \
                                                                \
float float_ ## name(size_t n, float *x)                        \
{                                                               \
    float arg = x[0];                                           \
    for (size_t i = 1; i < n; ++i) {                            \
        arg = (arg binop x[i]) ? arg : x[i];                    \
    }                                                           \
    return arg;                                                 \
}                                                               \
                                                                \
float float_ ## name ## _sse(size_t n, float *x)                \
{                                                               \
    float arg;                                                  \
                                                                \
    if (n < 8) {                                                \
        return float_ ## name(n, x);                            \
    }                                                           \
                                                                \
    float result4[4];                                           \
                                                                \
    __m128 arg4 = _mm_loadu_ps(x);                              \
    x += 4;                                                     \
    n -= 4;                                                     \
    while (n >= 4) {                                            \
        __m128 x4 = _mm_loadu_ps(x);                            \
        arg4 = _mm_ ## name ## _ps(arg4, x4);                   \
        x += 4;                                                 \
        n -= 4;                                                 \
    }                                                           \
    _mm_storeu_ps(result4, arg4);                               \
    arg = float_ ## name(4, result4);                           \
    if (n > 0) {                                                \
        arg = (arg binop x[0]) ? arg : x[0];                    \
    }                                                           \
    if (n > 1) {                                                \
        arg = (arg binop x[1]) ? arg : x[1];                    \
    }                                                           \
    if (n > 2) {                                                \
        arg = (arg binop x[2]) ? arg : x[2];                    \
    }                                                           \
    return arg;                                                 \
}                                                               \
                                                                \
float float_ ## name ## _avx(size_t n, float *x)                \
{                                                               \
    float arg;                                                  \
                                                                \
    if (n < 16) {                                               \
        return float_ ## name(n, x);                            \
    }                                                           \
                                                                \
    float result8[8];                                           \
                                                                \
    __m256 arg8 = _mm256_loadu_ps(x);                           \
    x += 8;                                                     \
    n -= 8;                                                     \
    while (n >= 8) {                                            \
        __m256 x8 = _mm256_loadu_ps(x);                         \
        arg8 = _mm256_ ## name ## _ps(arg8, x8);                \
        x += 8;                                                 \
        n -= 8;                                                 \
    }                                                           \
    _mm256_storeu_ps(result8, arg8);                            \
    arg = float_ ## name(8, result8);                           \
    if (n > 0) {                                                \
        arg = (arg binop x[0]) ? arg : x[0];                    \
    }                                                           \
    if (n > 1) {                                                \
        arg = (arg binop x[1]) ? arg : x[1];                    \
    }                                                           \
    if (n > 2) {                                                \
        arg = (arg binop x[2]) ? arg : x[2];                    \
    }                                                           \
    if (n > 3) {                                                \
        arg = (arg binop x[3]) ? arg : x[3];                    \
    }                                                           \
    if (n > 4) {                                                \
        arg = (arg binop x[4]) ? arg : x[4];                    \
    }                                                           \
    if (n > 5) {                                                \
        arg = (arg binop x[5]) ? arg : x[5];                    \
    }                                                           \
    if (n > 6) {                                                \
        arg = (arg binop x[6]) ? arg : x[6];                    \
    }                                                           \
    if (n > 7) {                                                \
        arg = (arg binop x[7]) ? arg : x[7];                    \
    }                                                           \
    return arg;                                                 \
}                                                               \

// Generate the functions float_{OP}_keep_nan, float_{OP}
// float_{OP}_sse and float_{OP}_avx.  These function compute
// the extreme value of a 1-D array of floats.
FLOAT_REDUCE_INFIX_BINOP(min, <)
FLOAT_REDUCE_INFIX_BINOP(max, >)
