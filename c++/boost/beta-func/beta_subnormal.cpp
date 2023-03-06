
#include <iostream>
#include <iomanip>
#include <cfenv>

#define BOOST_MATH_OVERFLOW_ERROR_POLICY errno_on_error

#include <boost/math/special_functions/beta.hpp>

using namespace std;

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
    double a = 12.5;
    double b = 1e-320;

    std::feclearexcept(FE_ALL_EXCEPT);

    double x = boost::math::beta(a, b);

    show_fp_exception_flags();

    std::cout << std::scientific << std::setw(24)
              << std::setprecision(17) << x << std::endl;

    return 0;
}
