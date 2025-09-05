// http://stackoverflow.com/questions/40156379/how-to-use-gsl-beta-distribution

#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

int main()
{
    double alpha = 10;
    double beta = 0.5;
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus2);
    double d = gsl_ran_beta(rng, alpha, beta);
    printf("d = %10.8f\n", d);
    return 0;
}