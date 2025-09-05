#include <stdio.h>
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"


int main(int argc, char *argv[])
{
    const gsl_rng_type *T;
    gsl_rng *r;

    int k;
    double x;

    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);

    for (k = 0; k < 5; ++k) {
        x = gsl_ran_gaussian(r, 1.0);
        printf("%13.8f\n", x);
    }

    gsl_rng_free(r);

    return 0;
}
