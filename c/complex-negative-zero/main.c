#include <stdio.h>
#include <complex.h>

int main(int argc, char *argv[])
{
    double complex z = CMPLX(0.125, -0.0);
    double complex m = z*2.5;
    // double complex u = z + CMPLX(1.0, 0.0);
    double complex u = z + 1.0;
    double complex w = z/u;
    double complex q = z / 1.25;
    printf("z = %10.8f + %10.8fi\n", creal(z), cimag(z));
    printf("m = %10.8f + %10.8fi\n", creal(m), cimag(m));
    printf("u = %10.8f + %10.8fi\n", creal(u), cimag(u));
    printf("w = %10.8f + %10.8fi\n", creal(w), cimag(w));
    printf("q = %10.8f + %10.8fi\n", creal(q), cimag(q));

    return 0;
}
