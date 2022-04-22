//
// An example of the use of the LAPACKE wrapper of DSYSV
// to compute the solution of Ax = b for symmetric A.
//

#include <stdio.h>
#include <lapacke.h>


void
printa(lapack_int n, double *a)
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%7.4f  ", a[i + j*n]);
        }
        printf("\n");
    }
}

void
printb(lapack_int n, double *b)
{
    for (int i = 0; i < n; ++i) {
        printf(" %7.4f", b[i]);
    }
    printf("\n");
}

int
main(int argc, char *argv[])
{
    double *a, *b;
    lapack_int *ipiv;
    lapack_int n = 4;
    lapack_int nrhs = 1;
    lapack_int lda = n;
    lapack_int ldb = n;
    lapack_int info;

    a = calloc(n*n, sizeof(double));
    if (a == NULL) {
        exit(-1);
    }
    b = calloc(n, sizeof(double));
    if (b == NULL) {
        free(a);
        exit(-1);
    }
    ipiv = calloc(n, sizeof(lapack_int));
    if (ipiv == NULL) {
        free(b);
        free(a);
        exit(-1);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            a[i + j*n] = 1 + i;
            a[j + i*n] = a[i + j*n];
        }
    }

    for (int i = 0; i < n; ++i) {
        b[i] = 1.0 + i*i;
    }

    printf("a:\n");
    printa(n, a);
    printf("b:");
    printb(n, b);

    info = LAPACKE_dsysv(LAPACK_COL_MAJOR, 'U', n, nrhs, a, lda, ipiv, b, ldb);
    printf("info = %d\n", info);

    if (info == 0) {
        printf("a:\n");
        printa(n, a);
        printf("b:");
        printb(n, b);
    }
    printf("ipiv:");
    for (int i = 0; i < n; ++i) {
        printf(" %5d", ipiv[i]);
    }
    printf("\n");

    free(b);
    free(a);

    return info;
}
