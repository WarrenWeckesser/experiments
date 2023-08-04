
#include <stdio.h>
#include <complex.h>
#include <math.h>

int main(int argc, char *argv[])
{
    double complex z = 0.5469835951544786 + 0.8371433277465153i;

    printf("z.real = %25.17e\n", creal(z));
    printf("z.imag = %25.17e\n", cimag(z));

    double complex w = clog(z);

    printf("w.real = %25.17e\n", creal(w));
    printf("w.imag = %25.17e\n", cimag(w));

    return 0;
}