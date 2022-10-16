
// Compile with
//   g++ -std=c++1z noncentralf.cpp -o noncentralf

#include <string>
#include <iostream>
#include <cfenv>
#include <cmath>
#include <boost/math/distributions/non_central_f.hpp>

using namespace std;
using boost::math::non_central_f;


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
    if (argc != 5 ) {
        printf("use: %s x v1 v2 lambda\n", argv[0]);
        return -1;
    }

    double x = std::stod(argv[1]);
    double v1 = std::stod(argv[2]);
    double v2 = std::stod(argv[3]);
    double lambda = std::stod(argv[4]);

    auto ncf = non_central_f(v1, v2, lambda);

    std::feclearexcept(FE_ALL_EXCEPT);
    double p = pdf(ncf, x);
    cout << "fp flags after pdf():";
    show_fp_exception_flags();

    std::feclearexcept(FE_ALL_EXCEPT);
    double c = cdf(ncf, x);
    cout << "fp flags after cdf():";
    show_fp_exception_flags();

    cout << "pdf: " << setw(24) << setprecision(16) << p << endl;
    cout << "cdf: " << setw(24) << setprecision(16) << c << endl;

    return 0;
}
