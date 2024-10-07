
#include <iostream>
#include <cfenv>
#include <boost/math/distributions/beta.hpp>

using namespace std;
using boost::math::beta_distribution;

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
    double p = 0.9;

    beta_distribution<> dist(a, b);

    std::feclearexcept(FE_ALL_EXCEPT);
    double x = quantile(dist, p);
    show_fp_exception_flags();

    cout << "quantile = " << scientific << setw(23) << setprecision(16) << x << endl;

    return 0;
}
