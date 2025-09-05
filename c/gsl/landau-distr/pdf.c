#include <stdio.h>
#include <gsl/gsl_randist.h>

int main()
{
    double x;
    for (x = -5.0; x <= 10.0; x += 0.01) {
        double p = gsl_ran_landau_pdf(x);
        printf("%15.12f %21.15e\n", x, p);
    }
    return 0;
}
