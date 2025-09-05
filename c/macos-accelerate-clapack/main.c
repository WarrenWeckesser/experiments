
#include <stdio.h>
#include <math.h>
#include <Accelerate/Accelerate.h>


int main(int argc, char *argv[])
{
    // With Fortran ordering, `a` is the transpose of how it appears here.
    double a[] = {1.0, 0.0, 0.0,
                  2.0, 5.0, 0.0,
                  3.0, 6.0, 1.0};
    double b[] = {10.0, 20.0, 5.0};
    int ipiv[3];
    int info;
    int n = 3;
    int nrhs = 1;
    int ldb = n;
    int k;
    double work[1000];
    int lwork = 1000;

    dsysv_("U", &n, &nrhs, a, &n, ipiv, b, &ldb, work, &lwork, &info);

    printf("info = %d\n", info);   
    for (k = 0; k < n; ++k) {
        printf("%10.5f\n", b[k]);
    }
    return 0;
}