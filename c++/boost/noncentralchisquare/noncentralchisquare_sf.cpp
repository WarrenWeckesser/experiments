
#include <cfenv>
#include <cmath>

#include <boost/math/distributions/non_central_chi_squared.hpp>

using boost::math::non_central_chi_squared;

#include <iostream>

using std::cout; using std::endl; using std::setw;
using std::scientific;
using std::fixed;
using std::setprecision;

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

int main()
{
    double x = 8.0;
    double nu = 1.0 - 1e-8;
    double lam = 1.0 + 1e-8;

    std::feclearexcept(FE_ALL_EXCEPT);
    double s = cdf(complement(non_central_chi_squared(nu, lam), x));
    show_fp_exception_flags();

    cout << scientific;
    cout << fixed << setw(8) << setprecision(4) << x << " ";
    cout << fixed << setw(8) << setprecision(4) << nu << " ";
    cout << fixed << setw(8) << setprecision(4) << lam << " ";
    cout << scientific << setprecision(17) << s << endl;
}
