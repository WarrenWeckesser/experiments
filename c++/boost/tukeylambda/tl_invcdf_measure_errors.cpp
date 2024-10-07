#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include <boost/multiprecision/cpp_bin_float.hpp>

#include "tukeylambda.h"

using boost::multiprecision::cpp_bin_float_100;
using boost::multiprecision::cpp_bin_float_50;


//
// Uses type T (e.g. cpp_bin_float_50, cpp_bin_float_100) internally
// for floating point calculations.
//
template<typename T>
double invcdf_mp(double p, double lam)
{
    T p_mp = p;
    T lam_mp = lam;
    T x_mp;
    if (lam == 0) {
        x_mp = log(p_mp) - log1p(-p_mp);
    }
    else {
        x_mp = (pow(p_mp, lam_mp) - pow(1 - p_mp, lam_mp))/lam_mp;
    }
    return static_cast<double>(x_mp);
}


inline double
relerr(double value, double ref)
{
    return abs(value - ref)/abs(ref);
}

int main()
{
    int order;
    double p = 0.05;
    std::vector<double> lambdas;

    double lambda = 1e-30;
    for (int k = 0; k < 620; ++k) {
        lambdas.push_back(lambda);
        lambdas.push_back(-lambda);
        lambda *= 1.125;
    }
    std::sort(lambdas.begin(), lambdas.end());

    printf("lambda        invcdf1      invcdf2      invcdf3      invcdft3     invcdft9     invcdft12    invcdft15    invcdfx\n");
    for (auto lam: lambdas) {
        double ref = invcdf_mp<cpp_bin_float_100>(p, lam);

        double invcdf1 = tukey_lambda_invcdf(p, lam);
        double invcdf1_re = relerr(invcdf1, ref);

        double invcdf2 = tukey_lambda_invcdf2(p, lam);
        double invcdf2_re = relerr(invcdf2, ref);

        double invcdf3 = tukey_lambda_invcdf3(p, lam);
        double invcdf3_re = relerr(invcdf3, ref);

        order = 3;
        double invcdft3 = tukey_lambda_invcdf_taylor(p, lam, order);
        double invcdft3_re = relerr(invcdft3, ref);

        order = 9;
        double invcdft9 = tukey_lambda_invcdf_taylor(p, lam, order);
        double invcdft9_re = relerr(invcdft9, ref);

        order = 12;
        double invcdft12 = tukey_lambda_invcdf_taylor(p, lam, order);
        double invcdft12_re = relerr(invcdft12, ref);

        order = 15;
        double invcdft15 = tukey_lambda_invcdf_taylor(p, lam, order);
        double invcdft15_re = relerr(invcdft15, ref);

        double invcdfx = tukey_lambda_invcdf_experimental(p, lam);
        double invcdfx_re = relerr(invcdfx, ref);

        printf("%11.6e %11.6e %11.6e %11.6e %11.6e %11.6e %11.6e %11.6e %11.6e\n",
               lam, invcdf1_re, invcdf2_re, invcdf3_re, invcdft3_re, invcdft9_re, invcdft12_re, invcdft15_re, invcdfx_re);
    }
}
