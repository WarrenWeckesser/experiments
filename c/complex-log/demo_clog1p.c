
#include <stdio.h>
#include "clog1p.h"
#include <complex.h>
#include <math.h>


double complex z_ref[][2] = {
    {CMPLX(-0.57113, -0.90337),
        CMPLX(3.4168883248419116e-06, -1.1275564209486122)},
    {CMPLX(3.42590243125101e-13, 6.718940751235093e-08), 
        CMPLX(3.4484745136597113e-13, 6.718940751232781e-08)},
    {CMPLX(-0.01524813, -0.173952),
        CMPLX(-2.228118716056777e-06, -0.1748418364650139)},
    {CMPLX(-0.011228922063957758, 0.14943813247359922),
        CMPLX(-4.499779370610757e-17, 0.15)},
    {CMPLX(3e-180, 2e-175), CMPLX(3e-180, 2e-175)}
};

int main(int argc, char *argv[])
{

    printf("Computing w = log1p(z):\n\n");
    for (int k = 0; k < sizeof(z_ref)/sizeof(z_ref[0]); ++k) {
        printf("z   = %25.16e + %25.16e j\n", creal(z_ref[k][0]), cimag(z_ref[k][0]));
        double complex w1 = clog1p(z_ref[k][0]);
        double complex w2 = clog1p_doubledouble(z_ref[k][0]);
        printf("w1  = %25.16e + %25.16e j\n", creal(w1), cimag(w1));
        printf("w2  = %25.16e + %25.16e j\n", creal(w2), cimag(w2));
        printf("ref = %25.16e + %25.16e j\n", creal(z_ref[k][1]), cimag(z_ref[k][1]));
        printf("\n");
    }

    return 0;
}
