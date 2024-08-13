
#include <cmath>
#include <cstdio>
#include "tukeylambda.h"

int main()
{
    double x = -3.96;
    double lam = 0.25;
    printf("x = %25.16e   lam = %25.16e\n", x, lam);

    double p = tukey_lambda_cdf(x, lam);
    printf("p  = %25.16e\n", p);

    double x1 = tukey_lambda_invcdf(p, lam);
    printf("x1 = %25.16e\n", x1);

    printf("-----\n");
    double pdf = tukey_lambda_pdf(x, lam);
    printf("pdf  = %25.16e\n", pdf);

    return std::isnan(p);
}
