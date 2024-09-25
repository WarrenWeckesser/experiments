
#include <cmath>
#include <cstdio>
#include "tukeylambda.h"

//
//   Reference values computed with mpmath
//       x    lambda      PDF/log PDF             CDF/SF
//    -3.999   0.25    1.5624999999567592e-11   3.906249999937244e-15
//                     -24.88214892033376       0.9999999999999961
//        -2  -1e-5    0.10499340481014478      0.11920528841881847
//                     -2.2538577421292825      0.8807947115811815
//     -5000  -0.25    3.263724198653559e-16    4.082918972515604e-13
//                     -35.65849255233456       0.9999999999995917
//

int main()
{
    double x = -3.999;
    double lam = 0.25;
    printf("x = %25.16e   lam = %25.16e\n", x, lam);

    double cdf = tukey_lambda_cdf(x, lam);
    printf("cdf = %25.16e\n", cdf);

    double sf = tukey_lambda_sf(x, lam);
    printf("sf  = %25.16e\n", sf);

    double invcdf = tukey_lambda_invcdf(cdf, lam);
    printf("invcdf  = %25.16e\n", invcdf);

    double invsf = tukey_lambda_invsf(sf, lam);
    printf("invsf   = %25.16e\n", invsf);

    printf("-----\n");
    double pdf = tukey_lambda_pdf(x, lam);
    printf("pdf     = %25.16e\n", pdf);
    double logpdf = tukey_lambda_logpdf(x, lam);
    printf("logpdf  = %25.16e\n", logpdf);
    double check = std::exp(logpdf);
    printf("check   = %25.16e\n", check);
}
