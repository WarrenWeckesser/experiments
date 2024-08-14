
#include <cmath>
#include <cstdio>
#include "tukeylambda.h"

int main()
{
    double x = -2.0;
    double lam = -1e-5;
    printf("x = %25.16e   lam = %25.16e\n", x, lam);

    double p = tukey_lambda_cdf(x, lam);
    printf("cdf = %25.16e\n", p);

    double x1 = tukey_lambda_invcdf(p, lam);
    printf("x1  = %25.16e\n", x1);

    printf("-----\n");
    double pdf = tukey_lambda_pdf(x, lam);
    printf("pdf     = %25.16e\n", pdf);
    double logpdf = tukey_lambda_logpdf(x, lam);
    printf("logpdf  = %25.16e\n", logpdf);
    double check = std::exp(logpdf);
    printf("check   = %25.16e\n", check);
    return std::isnan(p);
}
