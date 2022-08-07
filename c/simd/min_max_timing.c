
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "min_max.h"


int main(int argc, char *argv[])
{
    clock_t start, diff;
    float msec;

    if (argc != 2) {
        fprintf(stderr, "Specify the array size as the command line argument.\n");
        exit(-1);
    }
    long n = atol(argv[1]);
    printf("n = %ld\n", n);

    time_t seed = time(NULL);
    printf("seed = %ld\n", seed);
    srandom((unsigned) seed);

    float *x = calloc(n, sizeof(float));
    if (x == NULL) {
        fprintf(stderr, "calloc failed.\n");
        exit(-1);
    }
    for (long i = 0; i < n; ++i) {
        x[i] = random() / (float)((1L << 31) - 1);
    }

    start = clock();
    float min1 = float_min(n, x);
    diff = clock() - start;
    msec = 1000.0 * diff / CLOCKS_PER_SEC;
    printf("float_min:          %12.8f   %10.3f msec\n", min1, msec);

    start = clock();
    float min2 = float_min_sse(n, x);
    diff = clock() - start;
    msec = 1000.0 * diff / CLOCKS_PER_SEC;
    printf("float_min_sse:      %12.8f   %10.3f msec\n", min2, msec);

    start = clock();
    float min3 = float_min_avx(n, x);
    diff = clock() - start;
    msec = 1000.0 * diff / CLOCKS_PER_SEC;
    printf("float_min_avx:      %12.8f   %10.3f msec\n", min3, msec);

    start = clock();
    float min4 = float_min_keep_nan(n, x);
    diff = clock() - start;
    msec = 1000.0 * diff / CLOCKS_PER_SEC;
    printf("float_min_keep_nan: %12.8f   %10.3f msec\n", min4, msec);

    free(x);
    return 0;
}
