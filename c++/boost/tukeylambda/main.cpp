
#include <cmath>
#include <cstdio>
#include "tukeylambda.h"

int main()
{
    double x = 2500;
    double lam = -3;
    double p = tukey_lambda_sf(x, lam);
    printf("p  = %25.16e\n", p);
    double x1 = tukey_lambda_invsf(p, lam);

    printf("x1 = %25.16e\n", x1);
    return std::isnan(p);
}
