// XXX No one should use this code as an example of
//     *good* SIMD programming!

#include <math.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>


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
    while (n > 4) {                                             \
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
    while (n > 8) {                                             \
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


int main(int argc, char *argv[])
{
    size_t n = argc - 1;
    if (n == 0) {
        fprintf(stderr, "At least one number is required on the command line.\n");
        exit(-1);
    }
    float *x = malloc(n * sizeof(float));
    if (x == NULL) {
        fprintf(stderr, "malloc failed\n");
        exit(-1);
    }
    for (size_t i = 0; i < n; ++i) {
        x[i] = strtof(argv[i+1], NULL);
        // printf("x[%ld] = %8.3f\n", i, x[i]);
    }

    printf("min\n");

    float min1 = float_min(n, x);
    printf("float_min:          %12.8f\n", min1);

    float min2 = float_min_sse(n, x);
    printf("float_min_sse:      %12.8f\n", min2);

    float min3 = float_min_avx(n, x);
    printf("float_min_avx:      %12.8f\n", min3);

    float min4 = float_min_keep_nan(n, x);
    printf("float_min_keep_nan: %12.8f\n", min4);

    printf("max\n");

    float max1 = float_max(n, x);
    printf("float_max:          %12.8f\n", max1);

    float max2 = float_max_sse(n, x);
    printf("float_max_sse:      %12.8f\n", max2);

    float max3 = float_max_avx(n, x);
    printf("float_max_avx:      %12.8f\n", max3);

    float max4 = float_max_keep_nan(n, x);
    printf("float_max_keep_nan: %12.8f\n", max4);

    free(x);
    return 0;
}
