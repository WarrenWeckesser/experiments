#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <Accelerate/Accelerate.h>

#define NROWS 32
#define NCOLS 8

// We'll compute y = A x using cblas_zgemv.
double complex A[NROWS * NCOLS];
double complex x[NCOLS];
double complex y[NROWS];

int main(int argc, char *argv[])
{
    // Set the first value in all but the first and last rows of A to
    // CMPLX(inf, nan).  Other values in A are 0 (static initialized).
    // Similarly, x is initialized to all 0.
    // With this array A, we expect the number of nans in the output
    // to be NROWS - 2.
    for (int i = 1; i < NROWS - 1; ++i) {
        A[NCOLS*i] = CMPLX(INFINITY, NAN);
    }

    const double complex alpha = CMPLX(1.0, 0.0);
    const double complex beta  = 0.0;

    cblas_zgemv(
        CblasRowMajor,
        CblasNoTrans,
        NROWS,
        NCOLS,
        &alpha,
        A,
        NCOLS,
        x,
        1,
        &beta,
        y,
        1
    );

    int num_output_nans = 0;
    for (int i = 0; i < NROWS; ++i) {
        if (isnan(creal(y[i])) || isnan(cimag(y[i]))) {
            ++num_output_nans;
        }
    }
    printf("num_output_nans = %d  (expected %d)\n", num_output_nans, NROWS - 2);
    if (num_output_nans != NROWS - 2) {
        printf("\nThese two output values were not expected to contain nan:\n");
        int last = NROWS - 1;
        printf("y[0] = (%f,%f)\n", creal(y[0]), cimag(y[0]));
        printf("y[%d] = (%f,%f)\n", last, creal(y[last]), cimag(y[last]));
    }
}
