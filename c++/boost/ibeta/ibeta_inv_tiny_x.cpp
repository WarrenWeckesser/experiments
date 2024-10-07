
#include <iostream>
#include <iomanip>
#include <cfenv>
#include <boost/math/special_functions/beta.hpp>

using namespace std;
using namespace boost::math;

void show_fp_exception_flags()
{
    if (std::fetestexcept(FE_DIVBYZERO)) {
        cout << " FE_DIVBYZERO";
    }
    // FE_INEXACT is common and not interesting.
    // if (std::fetestexcept(FE_INEXACT)) {
    //     cout << " FE_INEXACT";
    // }
    if (std::fetestexcept(FE_INVALID)) {
        cout << " FE_INVALID";
    }
    if (std::fetestexcept(FE_OVERFLOW)) {
        cout << " FE_OVERFLOW";
    }
    if (std::fetestexcept(FE_UNDERFLOW)) {
        cout << " FE_UNDERFLOW";
    }
    cout << endl;
}

int main(int argc, char *argv[])
{
    double a = 14.208308325339239;
    double b = a;

    double p = 7.703145458496392e-307;  // Raises exception: Assertion `x >= 0' failed.
    // double p = 6.4898872103239473e-300;
    // double p = 7.8e-307;  // No flags set, returns 8.57354094063444939e-23
    // double p = 7.7e-307;  // FE_UNDERFLOW set, returns 0.0

    std::feclearexcept(FE_ALL_EXCEPT);

    double x = ibeta_inv(a, b, p);

    show_fp_exception_flags();

    std::cout << std::scientific << std::setw(24)
              << std::setprecision(17) << x << std::endl;

    return 0;
}
