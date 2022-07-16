
#include <stdio.h>
#include <stdlib.h>

#include "min_max.h"


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
