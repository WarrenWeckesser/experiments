
#include <stdio.h>
#include <gsl/gsl_sf_expint.h>

int main(int argc, char *argv[])
{
    double y;
    double x = 3.5;

    y = gsl_sf_expint_Ei(x);
    printf("gsl_sf_expint_Ei(%7.3f):     %15.10e\n", x, y);

    y = -gsl_sf_expint_Ei(-x);
    printf("-gsl_sf_expint_Ei(%7.3f):    %15.10e\n", -x, y);

    y = gsl_sf_expint_E1(x);
    printf("gsl_sf_expint_E1(%7.3f):     %15.10e\n", x, y);

    y = gsl_sf_expint_En(1, x);
    printf("gsl_sf_expint_En(1, %7.3f):  %15.10e\n", x, y);

    y = gsl_sf_expint_En(3, x);
    printf("gsl_sf_expint_En(3, %7.3f):  %15.10e\n", x, y);
    return 0;
}
