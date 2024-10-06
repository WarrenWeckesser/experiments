
#include <iostream>
#include <cmath>
#include <boost/math/special_functions/sin_pi.hpp>
#include <boost/math/special_functions/cos_pi.hpp>

using boost::math::sin_pi;
using boost::math::cos_pi;
using namespace std;

template <typename T>
T scipy_sinpi(T x)
{
    T s = 1.0;

    if (x < 0.0) {
        x = -x;
        s = -1.0;
    }

    T r = std::fmod(x, 2.0);
    if (r < 0.5) {
        return s * std::sin(M_PI * r);
    } else if (r > 1.5) {
        return s * std::sin(M_PI * (r - 2.0));
    } else {
        return -s * std::sin(M_PI * (r - 1.0));
    }
}

template <typename T>
T scipy_cospi(T x)
{
    if (x < 0.0) {
        x = -x;
    }

    T r = std::fmod(x, 2.0);
    if (r == 0.5) {
        // We don't want to return -0.0
        return 0.0;
    }
    if (r < 1.0) {
        return -std::sin(M_PI * (r - 0.5));
    } else {
        return std::sin(M_PI * (r - 1.5));
    }
}



int main()
{
    double xvals[] = {0.999, 0.999999999999, 0.99999999999999, 1.0, 23456789.01};

    cout << "   x                      sin_pi(x)                    cos_pi(x)" << endl;
    cout << "                          [scipy version]              [scipy version]" << endl;
    cout << "                          sin(pi*x)                    cos(pi*x)" << endl;
    for (const auto &x : xvals) {
        double sp = sin_pi(x);
        double scipy_sp = scipy_sinpi(x);
        double cp = cos_pi(x);
        double scipy_cp = scipy_cospi(x);
        cout << scientific
             << setprecision(15) << setw(21) << x
             << setprecision(17) << setw(26) << sp
             << setprecision(17) << setw(26) << cp
             << endl;
        cout << scientific
             << setw(21) << ""
             << setprecision(17) << setw(26) << scipy_sp
             << setprecision(17) << setw(26) << scipy_cp
             << endl;
        cout << scientific
             << setw(21) << ""
             << setprecision(17) << setw(26) << sin(M_PI*x)
             << setprecision(17) << setw(26) << cos(M_PI*x)
             << endl;
    }
}
