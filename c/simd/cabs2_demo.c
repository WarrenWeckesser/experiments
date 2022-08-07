
#include <stdio.h>
#include <stdlib.h>

#include "cabs2.h"


void print_float_array(size_t n, float *a)
{
    for (size_t k = 0; k < n; ++k) {
        printf("%9.3f ", a[k]);
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    size_t m = argc - 1;
    if (m < 2) {
        fprintf(stderr, "At least two numbers are required on the command line.\n");
        exit(-1);
    }
    if (m & 1) {
        fprintf(stderr, "The number of values must be even.  The last value is will be ignored.\n");
        m -= 1;
    }
    size_t n = m / 2;
    float *mem = malloc((3*n) * sizeof(float));
    if (mem == NULL) {
        fprintf(stderr, "malloc failed\n");
        exit(-1);
    }
    float *x = mem;
    float *out = mem + m;

    for (size_t i = 0; i < m; ++i) {
        x[i] = strtof(argv[i+1], NULL);
    }
    for (size_t i = 0; i < m; i += 2) {
        printf("z[%ld] = %8.3f + %8.3fi\n", i/2, x[i], x[i+1]);
    }

    cabs2f(n, x, out);
    print_float_array(n, out);

    for (size_t k = 0; k < n; ++k) {
        out[k] = 0.0f;
    }

    cabs2f_sse(n, x, out);
    print_float_array(n, out);

    return 0;
}
