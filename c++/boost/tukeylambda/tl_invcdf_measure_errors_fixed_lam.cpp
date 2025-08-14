#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    if (ref == 0.0) {
        if (value == ref) {
            return 0.0;
        }
        else {
            return std::numeric_limits<double>::infinity();
        }
    }
    return abs(value - ref)/abs(ref);
}

int main(int argc, char *argv[])
{
    int order;
    double lam;
    if (argc > 1) {
        lam = strtold(argv[1], nullptr);
    }
    else {
        fprintf(stderr, "Expected one argument: lambda\n");
        exit(-1);
    }
    std::vector<double> pvalues;

    for (int k = 1; k < 2000; ++k) {
        pvalues.push_back(k/2000.0);
    }
    for (int k = -4000; k <= 4000; ++k) {
        pvalues.push_back(0.5 + (2*k + 1)/1000001.0);
    }
    for (int k = -235; k < 0; ++k) {
        pvalues.push_back(std::exp(k));
    }
    for (int k = -36; k < 0; ++k) {
        pvalues.push_back(-std::expm1(k));
    }
    std::sort(pvalues.begin(), pvalues.end());

    printf("p                       invcdf1      invcdf2      invcdf3a     invcdf3b     invcdft3     invcdft9     invcdft12    invcdft15    invcdfx invcdftp\n");
    for (auto p: pvalues) {
        double ref = invcdf_mp<cpp_bin_float_100>(p, lam);

        double invcdf1 = tukey_lambda_invcdf(p, lam);
        double invcdf1_re = relerr(invcdf1, ref);

        double invcdf2 = tukey_lambda_invcdf2(p, lam);
        double invcdf2_re = relerr(invcdf2, ref);

        double invcdf3a = tukey_lambda_invcdf3a(p, lam);
        double invcdf3a_re = relerr(invcdf3a, ref);

        double invcdf3b = tukey_lambda_invcdf3b(p, lam);
        double invcdf3b_re = relerr(invcdf3b, ref);

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

        order = 13;
        double invcdftp = tukey_lambda_invcdf_taylor_p(p, lam, order);
        double invcdftp_re = relerr(invcdftp, ref);

        printf("%22.15e %11.6e %11.6e %11.6e %11.6e %11.6e %11.6e %11.6e %11.6e %11.6e %11.6e\n",
               p, invcdf1_re, invcdf2_re, invcdf3a_re, invcdf3b_re, invcdft3_re, invcdft9_re, invcdft12_re, invcdft15_re, invcdfx_re, invcdftp_re);
    }
}
