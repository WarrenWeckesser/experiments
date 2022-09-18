//
// func.c
//

#include <math.h>

// C implementation of the function to be integrated.
// This function will be wrapped as a LowLevelCallable that can
// be given to scipy.integrate.quad.

double func(int n, double *t, void *xab) {
    double x = ((double *) xab)[0];
    double a = ((double *) xab)[1];
    double b = ((double *) xab)[2];

    return exp(*t * x) * pow(*t, a - 1) * pow(1 - *t, b - a - 1);
}