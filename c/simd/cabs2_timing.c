
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

#include "cabs2.h"


void compare_arrays(char *name, size_t n, float* out1, float *out2, float *x)
{
    for (size_t i = 0; i < n; ++i) {
        if (out2[i] != out1[i]) {
            fprintf(stderr, "%s: output does not match cabs2f exactly.\n", name);
            fprintf(stderr, "z = %18.10f + %18.10f i\n", x[2*i], x[2*i + 1]);
            double xx = x[2*i];
            double yy = x[2*i+1];
            float ref = (float) (xx*xx + yy*yy);
            fprintf(stderr, "'true' value:     %13.10f\n", ref);
            fprintf(stderr, "out1[%7ld]:    %13.10f (%e)\n", i, out1[i], fabsf((out1[i] - ref)/ref));
            fprintf(stderr, "out2[%7ld]:    %13.10f (%e)\n", i, out2[i], fabsf((out2[i] - ref)/ref));
            fprintf(stderr, "\n");
            break;
        }
    }
}

int main(int argc, char *argv[])
{
    clock_t start, diff;
    float msec;

    if (argc < 2) {
        fprintf(stderr, "Specify the number of complex samples to use.\n");
        exit(-1);
    }
    if (argc > 2) {
        fprintf(stderr, "Only one argument is accepted, which is "
                        "the number of samples to use.\n");
        exit(-1);
    }
    long n = atol(argv[1]);
    n = n - (n % 2);
    printf("n = %ld\n", n);

    time_t seed = time(NULL);
    printf("seed = %ld\n", seed);
    srandom((unsigned) seed);

    float *x = aligned_alloc(sizeof(__m256), 2*n*sizeof(float));
    if (x == NULL) {
        fprintf(stderr, "aligned_alloc failed.\n");
        exit(-1);
    }
    for (long i = 0; i < 2*n; ++i) {
        x[i] = random() / (float)((1L << 31) - 1);
    }

    float *out1 = aligned_alloc(sizeof(__m256), n*sizeof(float));
    if (out1 == NULL) {
        fprintf(stderr, "aligned_alloc failed.\n");
        free(x);
        exit(-1);
    }

    float *out2 = aligned_alloc(sizeof(__m256), n*sizeof(float));
    if (out2 == NULL) {
        fprintf(stderr, "aligned_alloc failed.\n");
        free(x);
        free(out1);
        exit(-1);
    }

    float *out3 = aligned_alloc(sizeof(__m256), n*sizeof(float));
    if (out3 == NULL) {
        fprintf(stderr, "aligned_alloc failed.\n");
        free(x);
        free(out1);
        free(out2);
        exit(-1);
    }

    float *out4 = aligned_alloc(sizeof(__m256), n*sizeof(float));
    if (out4 == NULL) {
        fprintf(stderr, "aligned_alloc failed.\n");
        free(x);
        free(out1);
        free(out2);
        free(out3);
        exit(-1);
    }

    start = clock();
    cabs2f(n, x, out1);
    diff = clock() - start;
    msec = 1000.0 * diff / CLOCKS_PER_SEC;
    printf("cabs2f:          %10.3f msec\n", msec);

    start = clock();
    cabs2f_sse(n, x, out2);
    diff = clock() - start;
    msec = 1000.0 * diff / CLOCKS_PER_SEC;
    printf("cabs2f_sse:      %10.3f msec\n", msec);

    start = clock();
    cabs2f_avx(n, x, out3);
    diff = clock() - start;
    msec = 1000.0 * diff / CLOCKS_PER_SEC;
    printf("cabs2f_avx:      %10.3f msec\n", msec);

    start = clock();
    cabs2f_avx_v2(n, x, out4);
    diff = clock() - start;
    msec = 1000.0 * diff / CLOCKS_PER_SEC;
    printf("cabs2f_avx_v2:   %10.3f msec\n", msec);

    printf("\n");
    compare_arrays("cabs2f_sse", n, out1, out2, x);
    compare_arrays("cabs2f_avx", n, out1, out3, x);
    compare_arrays("cabs2f_avx_v2", n, out1, out4, x);

    free(out1);
    free(out2);
    free(out3);
    free(out4);
    free(x);
    return 0;
}
