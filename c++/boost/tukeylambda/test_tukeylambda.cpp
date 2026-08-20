#include <cstdint>
#include "checkit.h"
#include "tukeylambda.h"

//
// Reference values were computed with the Python library mpmath.
//
// Values followed by the comment '// WA' were also verified with Wolfram Alpha.
// e.g. for lambda = -1.5 and x = 10e10:
//     PDF[TukeyLambdaDistribution[-3/2], 10000000000]
//     CDF[TukeyLambdaDistribution[-3/2], 10000000000]
//     SurvivalFunction[TukeyLambdaDistribution[-3/2], 10000000000]
// For the inverse CDF, an input such a p=0.4999 is converted to its integer ratio
// to ensure that the input value used by Wolfram Alpah is exactly the same as that
// used here.  In Python, that ratio can be found by using the `.as_integer_ratio()`
// method:
//     In [4]: (0.4999).as_integer_ratio()
//     Out[4]: (2251349453722511, 4503599627370496)
// So to find the reference value of e.g. inverse_cdf(p=0.4999, lam=-5.0) with
// Wolfram Alpha, use:
//     InverseCDF[TukeyLambdaDistribution[-5], 2251349453722511/4503599627370496]
//

struct TestCase {
    double x;
    double lam;
    double ref;
};

struct TestCaseP {
    double p;
    double lam;
    double ref;
};

void test_pdf(CheckIt& test)
{
    TestCase cases[] = {
        {1e3, -0.25, 1.0037634844565678e-12},  // WA
        {1e6, -0.25, 1.0239795202457577e-27},  // WA
        {1e9, -0.25, 1.0239999795200002e-42},  // WA

        {1e6,  -0.5, 7.999952000191999e-18},   // WA
        {1e8,  -0.5, 7.999999520000019e-24},   // WA
        {1e10, -0.5, 7.9999999952e-30},        // WA

        {1e5,  -1.5, 2.3614372292940414e-09},
        {1e10, -1.5, 1.0960942551361985e-17},  // WA
        {1e14, -1.5, 2.3614634870724426e-24},  // WA
        {1e32, -1.5, 2.361463487072469e-54}
    };

    for (auto [x, lam, ref]: cases) {
        double pdf = tukey_lambda_pdf(x, lam);
        double rtol = 5e-15;
        assert_close_fp(test, pdf, ref, rtol, "PDF not close to reference");
    }
}

void test_cdf(CheckIt& test)
{
    TestCase cases[] = {
        {1e3,  -0.25, 0.9999999997480553},  // WA
        {1e6,  -0.25, 1.0},
        {1e9,  -0.25, 1.0},

        {1e6,  -0.5,  0.999999999996},
        {1e8,  -0.5,  0.9999999999999996},  // WA
        {1e10, -0.5,  1.0},

        {1e5,  -1.5,  0.9996457820520762},  // WA
        {1e10, -1.5,  0.9999998355858617},  // WA
        {1e14, -1.5,  0.9999999996457805}   // WA
    };

    for (auto [x, lam, ref]: cases) {
        double pdf = tukey_lambda_cdf(x, lam);
        double rtol = 5e-15;
        assert_close_fp(test, pdf, ref, rtol, "CDF not close to reference");
    }
}

void test_sf(CheckIt& test)
{
    TestCase cases[] = {
        {1e3,  -0.25, 2.5194463459891463e-10},  // WA
        {1e6,  -0.25, 2.559959040409597e-22},   // WA
        {1e9,  -0.25, 2.5599999590400004e-34},  // WA

        {1e6,  -0.5,  3.9999840000479995e-12},
        {1e8,  -0.5,  3.9999998400000047e-16},
        {1e10, -0.5,  3.9999999984e-20},        // WA

        {1e5,  -1.5,  0.0003542179479237911},
        {1e8,  -1.5,  3.54219521486553e-06},
        {1e10, -1.5,  1.644141382813907e-07},   // WA
        {1e14, -1.5,  3.5421952306086877e-10},
        {1e23, -1.5,  3.542195230608704e-16},   // WA
        {1e32, -1.5,  3.5421952306087036e-22}   // WA
    };

    for (auto [x, lam, ref]: cases) {
        double pdf = tukey_lambda_sf(x, lam);
        double rtol = 5e-15;
        assert_close_fp(test, pdf, ref, rtol, "SF not close to reference");
    }
}

void test_invcdf(CheckIt& test)
{
    TestCaseP cases[] = {
        {0.50001,       -15.0,    1.3107200237617578},
        {1e-32,          -5.0,   -1.9999999999999995e+159},  // WA
        {0.4999,         -5.0,   -0.012800003583999107},     // WA
        {0.99999999999,  -5.0,    1.9999991725964955e+54},   // WA
        {1e-80,          -3.5,   -2.8571428571428575e+279},
        {0.49999999,     -3.0,   -3.1999999983156655e-07},
        {0.500000001,    -3.0,    3.199999909497819e-08},
        {0.500000001,    -0.125,  4.362030807294371e-09},
        {0.5001,         -0.125,  0.000436203100018042},     // WA
        {0.4999999,       1e-13, -4.0000000001147987e-07},
        {0.500001,        1e-13,  4.000000000120079e-06},
        {0.500005,        1e-10,  1.9999999999411395e-05},
        {0.49999999,      8.0,   -1.5624999991775734e-10},   // WA
        {0.500000001,     8.0,    1.562499955809482e-11},
        {0.51,            8.0,    0.00015668767501000015},
        {0.575,          12.5,    7.742769011540703e-05}     // WA
    };

    for (auto [p, lam, ref]: cases) {
        double x = tukey_lambda_invcdf3a(p, lam);
        double rtol = 5e-15;
        assert_close_fp(test, x, ref, rtol, "INVCDF not close to reference");
    }
}

int main(void)
{
    CheckIt test = CheckIt(std::cerr);
    test_pdf(test);
    test_cdf(test);
    test_sf(test);
    test_invcdf(test);

    return test.print_summary("SUMMARY: ");
}
