
#include <iostream>
#include <cmath>
#include <boost/math/special_functions/sin_pi.hpp>
#include <boost/math/special_functions/cos_pi.hpp>

using boost::math::sin_pi;
using boost::math::cos_pi;
using namespace std;


int main()
{
    double xvals[] = {1.0, 23456789.01};

    cout << "   x           sin_pi(x)                 cos_pi(x)" << endl;
    cout << "               sin(pi*x)                 cos(pi*x)" << endl;
    for (const auto &x : xvals) {
        double sp = sin_pi(x);
        double cp = cos_pi(x);
        cout << scientific
             << setprecision(4) << setw(11) << x
             << setprecision(17) << setw(26) << sp
             << setprecision(17) << setw(26) << cp
             << endl;
        cout << scientific
             << setw(11) << ""
             << setprecision(17) << setw(26) << sin(M_PI*x)
             << setprecision(17) << setw(26) << cos(M_PI*x)
             << endl;
    }
}
