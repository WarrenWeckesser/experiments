
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
    if (argc != 4) {
        std::cout << "use: " << argv[0] << " a b p\n";
        return -1;
    }

    double a = std::stod(argv[1]);
    double b = std::stod(argv[2]);
    double p = std::stod(argv[3]);
    if (p <= 0 || p >= 1) {
        std::cerr << "must have 0 < p < 1" << std::endl;
        return -1;
    }

    cout << "a = " << a << endl;
    cout << "b = " << b << endl;
    beta_distribution<> dist(a, b);

    std::feclearexcept(FE_ALL_EXCEPT);
    double x = quantile(dist, p);
    show_fp_exception_flags();

    std::feclearexcept(FE_ALL_EXCEPT);
    double y = quantile(complement(dist, p));
    show_fp_exception_flags();

    //cout << fixed << setw(11) << setprecision(8) << x << " ";
    cout << scientific << setw(23) << setprecision(16) << x << " ";
    cout << scientific << setw(23) << setprecision(16) << y << " ";
    cout << endl;

    return 0;
}
