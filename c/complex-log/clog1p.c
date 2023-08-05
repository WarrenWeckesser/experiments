
#include <complex.h>
#include <math.h>

double complex
clog1p(double complex z)
{
    double complex w;

    if (isnan(creal(z)) || isnan(cimag(z))) {
        w = CMPLX(NAN, NAN);
    }
    else {
        double _Complex u = z + 1.0;
        if (creal(u) == 1.0 && cimag(u) == 0.0) {
            // z + 1 == 1
            w = z;
        }
        else {
            if (creal(u) - 1.0 == creal(z)) {
                // u - 1 == z
                w = clog(u);
            }
            else {
                w = clog(u) * (z / (u - 1.0));
            }
        }
    }
    return w;
}