//
// An example of the use of the LAPACKE wrapper of ZGEEV
// to compute the eigenvalues of complex matrix.
//

#include <stdio.h>
#include <complex.h>
#include <lapacke.h>


void
printa(lapack_int n, lapack_complex_double *a)
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            lapack_complex_double v = a[i + j*n];
            printf("%7.4f + %7.4fj  ", creal(v), cimag(v));
        }
        printf("\n");
    }
}


int
main(int argc, char *argv[])
{
    lapack_complex_double *a, *w, *vr;
    lapack_int n = 4;
    lapack_int status;

    a = calloc(n*n, sizeof(lapack_complex_double));
    if (a == NULL) {
        exit(-1);
    }
    w = calloc(n, sizeof(lapack_complex_double));
    if (w == NULL) {
        exit(-1);
    }
    vr = calloc(n*n, sizeof(lapack_complex_double));
    if (vr == NULL) {
        exit(-1);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i + j*n] = lapack_make_complex_double(i+j, 1 + i);
        }
    }
    a[0] = lapack_make_complex_double(0.0, 0.0);
    a[n*n-1] = lapack_make_complex_double(0.0, 0.0);
    printa(n, a);

    // XXX Check row order convention.
    status = LAPACKE_zgeev(LAPACK_ROW_MAJOR, 'N', 'V', n, a, n, w, NULL, n, vr, n);
    printf("status = %d\n", status);

    if (status == 0) {
        printf("Eigenvalues:\n");
        for (int i = 0; i < n; ++i) {
            printf("%22.17lf + %22.17lfj\n", creal(w[i]), cimag(w[i]));
        }
    }

    free(vr);
    free(w);
    free(a);

    return status;
}
