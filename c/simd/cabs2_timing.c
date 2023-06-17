
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "cabs2.h"


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

    float *x = calloc(2*n, sizeof(float));
    if (x == NULL) {
        fprintf(stderr, "calloc failed.\n");
        exit(-1);
    }
    for (long i = 0; i < 2*n; ++i) {
        x[i] = random() / (float)((1L << 31) - 1);
    }

    float *out = malloc(n*sizeof(float));
    if (out == NULL) {
        fprintf(stderr, "malloc failed.\n");
        free(x);
        exit(-1);
    }

    start = clock();
    cabs2f(n, x, out);
    diff = clock() - start;
    msec = 1000.0 * diff / CLOCKS_PER_SEC;
    printf("cabs2:          %10.3f msec\n", msec);

    start = clock();
    cabs2f_sse(n, x, out);
    diff = clock() - start;
    msec = 1000.0 * diff / CLOCKS_PER_SEC;
    printf("cabs2_sse:      %10.3f msec\n", msec);

    start = clock();
    cabs2f_avx(n, x, out);
    diff = clock() - start;
    msec = 1000.0 * diff / CLOCKS_PER_SEC;
    printf("cabs2_avx:      %10.3f msec\n", msec);

    free(out);
    free(x);
    return 0;
}
